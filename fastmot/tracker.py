from collections import OrderedDict
import itertools
import logging
import numpy as np
import numba as nb
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils.rect import as_rect, to_tlbr, iom


LOGGER = logging.getLogger(__name__)
CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of chi-square distribution
INF_COST = 1e5


class MultiTracker:
    """
    Uses optical flow and kalman filter to track multiple objects and
    associates detections to tracklets based on motion and appearance.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    dt : float
        Time interval in seconds between each frame.
    metric : string
        Feature distance metric to associate tracklets. Usually
        `euclidean` or `cosine`.
    config : Dict
        Tracker hyperparameters.
    """

    def __init__(self, size, dt, metric, config):
        self.size = size
        self.metric = metric
        self.max_age = config['max_age']
        self.age_factor = config['age_factor']
        self.motion_weight = config['motion_weight']
        self.max_feat_cost = config['max_feat_cost']
        self.max_reid_cost = config['max_reid_cost']
        self.iou_thresh = config['iou_thresh']
        self.duplicate_iou = config['duplicate_iou']
        self.conf_thresh = config['conf_thresh']
        self.lost_buf_size = config['lost_buf_size']

        self.next_id = 1
        self.tracks = {}
        self.lost = OrderedDict()
        self.kf = KalmanFilter(dt, config['kalman_filter'])
        self.flow = Flow(self.size, config['flow'])
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.flow_bboxes = {}
        self.homography = None

    def initiate(self, frame, detections):
        """
        Initializes the tracker from detections in the first frame.
        Parameters
        ----------
        frame : ndarray
            Initial frame.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        """
        if self.tracks:
            self.tracks.clear()
        self.flow.initiate(frame)
        for det in detections:
            state = self.kf.initiate(det.tlbr)
            new_trk = Track(0, self.next_id, det.tlbr, state, det.label)
            self.tracks[self.next_id] = new_trk
            LOGGER.debug('Detected: %s', new_trk)
            self.next_id += 1

    def track(self, frame):
        """
        Convenience function that combines `compute_flow` and `step_kalman_filter`.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        self.compute_flow(frame)
        self.step_kalman_filter()

    def compute_flow(self, frame):
        """
        Computes optical flow to estimate tracklet positions and camera motion.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        active_tracks = [track for track in self.tracks.values() if track.active]
        self.flow_bboxes, self.homography = self.flow.predict(frame, active_tracks)
        if self.homography is None:
            # clear tracks when camera motion cannot be estimated
            self.tracks.clear()

    def step_kalman_filter(self):
        """
        Performs kalman filter prediction and update from flow measurements.
        The function should be called after `compute_flow`.
        """
        for trk_id, track in list(self.tracks.items()):
            mean, cov = track.state
            mean, cov = self.kf.warp(mean, cov, self.homography)
            mean, cov = self.kf.predict(mean, cov)
            if trk_id in self.flow_bboxes:
                flow_tlbr = self.flow_bboxes[trk_id]
                # give large flow uncertainty for occluded tracks
                # usually these with high age and low inlier ratio
                std_multiplier = max(self.age_factor * track.age, 1) / track.inlier_ratio
                mean, cov = self.kf.update(mean, cov, flow_tlbr, MeasType.FLOW, std_multiplier)
            next_tlbr = as_rect(mean[:4])
            track.update(next_tlbr, (mean, cov))
            if iom(next_tlbr, self.frame_rect) < 0.5:
                if track.confirmed:
                    LOGGER.info('Out: %s', track)
                    self._mark_lost(trk_id)
                else:
                    del self.tracks[trk_id]

    def update(self, frame_id, detections, embeddings):
        """
        Associates detections to tracklets based on motion and feature embeddings.
        Parameters
        ----------
        frame_id : int
            The next frame ID.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        embeddings : ndarray
            NxM matrix of N extracted embeddings with dimension M.
        """
        det_ids = list(range(len(detections)))
        confirmed = [trk_id for trk_id, track in self.tracks.items() if track.confirmed]
        unconfirmed = [trk_id for trk_id, track in self.tracks.items() if not track.confirmed]

        # association with motion and embeddings
        cost = self._matching_cost(confirmed, detections, embeddings)
        matches1, u_trk_ids1, u_det_ids = self._linear_assignment(cost, confirmed, det_ids)

        # 2nd association with IoU
        active = [trk_id for trk_id in u_trk_ids1 if self.tracks[trk_id].active]
        u_trk_ids1 = [trk_id for trk_id in u_trk_ids1 if not self.tracks[trk_id].active]
        u_detections = detections[u_det_ids]
        cost = self._iou_cost(active, u_detections)
        matches2, u_trk_ids2, u_det_ids = self._linear_assignment(cost, active, u_det_ids, True)

        # 3rd association with unconfirmed tracks
        u_detections = detections[u_det_ids]
        cost = self._iou_cost(unconfirmed, u_detections)
        matches3, u_trk_ids3, u_det_ids = self._linear_assignment(cost, unconfirmed,
                                                                  u_det_ids, True)

        # re-id with lost tracks
        lost_ids = list(self.lost.keys())
        u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]
        u_detections, u_embeddings = detections[u_det_ids], embeddings[u_det_ids]
        cost = self._reid_cost(u_detections, u_embeddings)
        reid_matches, _, u_det_ids = self._linear_assignment(cost, lost_ids, u_det_ids)

        matches = itertools.chain(matches1, matches2, matches3)
        u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)
        updated, aged = [], []

        # update matched tracks
        for trk_id, det_id in matches:
            track = self.tracks[trk_id]
            det = detections[det_id]
            mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
            next_tlbr = as_rect(mean[:4])
            track.update(next_tlbr, (mean, cov), embeddings[det_id])
            if track.hits == 1:
                LOGGER.info('Found: %s', track)
            if iom(next_tlbr, self.frame_rect) < 0.5:
                LOGGER.info('Out: %s', track)
                self._mark_lost(trk_id)
            else:
                updated.append(trk_id)

        # reactivate matched lost tracks
        for trk_id, det_id in reid_matches:
            track = self.lost[trk_id]
            det = detections[det_id]
            LOGGER.info('Re-identified: %s', track)
            state = self.kf.initiate(det.tlbr)
            track.reactivate(frame_id, det.tlbr, state, embeddings[det_id])
            self.tracks[trk_id] = track
            del self.lost[trk_id]
            updated.append(trk_id)

        # clean up lost tracks
        for trk_id in u_trk_ids:
            track = self.tracks[trk_id]
            if not track.confirmed:
                LOGGER.debug('Unconfirmed: %s', track)
                del self.tracks[trk_id]
                continue
            track.mark_missed()
            if track.age > self.max_age:
                LOGGER.info('Lost: %s', track)
                self._mark_lost(trk_id)
            else:
                aged.append(trk_id)

        # register new detections
        for det_id in u_det_ids:
            det = detections[det_id]
            state = self.kf.initiate(det.tlbr)
            new_trk = Track(frame_id, self.next_id, det.tlbr, state, det.label)
            self.tracks[self.next_id] = new_trk
            LOGGER.debug('Detected: %s', new_trk)
            updated.append(self.next_id)
            self.next_id += 1

        # remove duplicate tracks
        self._remove_duplicate(updated, aged)

    def _mark_lost(self, trk_id):
        self.lost[trk_id] = self.tracks[trk_id]
        if len(self.lost) > self.lost_buf_size:
            self.lost.popitem(last=False)
        del self.tracks[trk_id]

    def _matching_cost(self, trk_ids, detections, embeddings):
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections)))

        features = [self.tracks[trk_id].smooth_feature for trk_id in trk_ids]
        cost = cdist(features, embeddings, self.metric)
        for i, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            motion_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            cost[i] = self._fuse_motion(cost[i], motion_dist, track.label, detections.label,
                                        self.max_feat_cost, self.motion_weight)
        return cost

    def _iou_cost(self, trk_ids, detections):
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections)))

        # make sure associated pair has the same class label
        trk_labels = np.array([self.tracks[trk_id].label for trk_id in trk_ids])
        trk_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        det_bboxes = detections.tlbr
        ious = bbox_overlaps(trk_bboxes, det_bboxes)
        ious = self._gate_cost(ious, trk_labels, detections.label, self.iou_thresh, True)
        return ious

    def _reid_cost(self, detections, embeddings):
        if len(self.lost) == 0 or len(detections) == 0:
            return np.empty((len(self.lost), len(detections)))

        trk_labels = np.array([track.label for track in self.lost.values()])
        features = [track.smooth_feature for track in self.lost.values()]
        cost = cdist(features, embeddings, self.metric)
        cost = self._gate_cost(cost, trk_labels, detections.label, self.max_reid_cost, False)
        return cost

    def _remove_duplicate(self, updated, aged):
        if len(updated) == 0 or len(aged) == 0:
            return

        updated_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in updated])
        aged_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in aged])

        ious = bbox_overlaps(updated_bboxes, aged_bboxes)
        idx = np.where(ious >= self.duplicate_iou)
        dup_ids = set()
        for row, col in zip(*idx):
            updated_id, aged_id = updated[row], aged[col]
            if self.tracks[updated_id].start_frame <= self.tracks[aged_id].start_frame:
                dup_ids.add(aged_id)
            else:
                dup_ids.add(updated_id)
        for trk_id in dup_ids:
            LOGGER.debug('Duplicate: %s', self.tracks[trk_id])
            del self.tracks[trk_id]

    @staticmethod
    def _linear_assignment(cost, trk_ids, det_ids, maximize=False):
        rows, cols = linear_sum_assignment(cost, maximize)
        unmatched_rows = list(set(range(cost.shape[0])) - set(rows))
        unmatched_cols = list(set(range(cost.shape[1])) - set(cols))
        unmatched_trk_ids = [trk_ids[row] for row in unmatched_rows]
        unmatched_det_ids = [det_ids[col] for col in unmatched_cols]
        matches = []
        if not maximize:
            for row, col in zip(rows, cols):
                if cost[row, col] < INF_COST:
                    matches.append((trk_ids[row], det_ids[col]))
                else:
                    unmatched_trk_ids.append(trk_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        else:
            for row, col in zip(rows, cols):
                if cost[row, col] > 0:
                    matches.append((trk_ids[row], det_ids[col]))
                else:
                    unmatched_trk_ids.append(trk_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        return matches, unmatched_trk_ids, unmatched_det_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fuse_motion(cost, motion_dist, label, det_labels, max_cost, weight):
        gate = (cost > max_cost) | (motion_dist > CHI_SQ_INV_95) | (label != det_labels)
        cost = (1 - weight) * cost + weight * motion_dist
        cost[gate] = INF_COST
        return cost

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _gate_cost(cost, trk_labels, det_labels, thresh, maximize):
        for i in nb.prange(len(cost)):
            if maximize:
                gate = (cost[i] < thresh) | (trk_labels[i] != det_labels)
                cost[i][gate] = 0
            else:
                gate = (cost[i] > thresh) | (trk_labels[i] != det_labels)
                cost[i][gate] = INF_COST
        return cost
