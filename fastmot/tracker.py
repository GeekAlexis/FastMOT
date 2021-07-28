from collections import OrderedDict
import itertools
import logging
import numpy as np
import numba as nb
from scipy.optimize import linear_sum_assignment
from cython_bbox import bbox_overlaps

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils.distance import cdist
from .utils.rect import as_rect, to_tlbr, ios, iom


LOGGER = logging.getLogger(__name__)
CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of chi-square distribution
INF_COST = 1e5


class MultiTracker:
    """
    Uses optical flow and Kalman filter to track multiple objects and
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
        Tracker parameters.
    """
    _hist_tracks = OrderedDict()

    def __init__(self, size, metric, config):
        self.size = size
        self.metric = metric
        self.max_age = config['max_age']
        self.age_penalty = config['age_penalty']
        self.age_weight = config['age_weight']
        self.motion_weight = config['motion_weight']
        self.max_feat_cost = config['max_feat_cost']
        self.max_reid_cost = config['max_reid_cost']
        self.iou_thresh = config['iou_thresh']
        self.duplicate_iou = config['duplicate_iou']
        self.occlusion_thresh = config['occlusion_thresh']
        self.conf_thresh = config['conf_thresh']
        self.confirm_hits = config['confirm_hits']
        self.history_size = config['history_size']

        self.tracks = {}
        self.kf = KalmanFilter(config['kalman_filter'])
        self.flow = Flow(self.size, config['flow'])
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.klt_bboxes = {}
        self.homography = None

    def reset(self, dt):
        """
        Reset the tracker for new input context.
        Parameters
        ----------
        dt : float
            Time interval in seconds between each frame.
        """
        self.kf.reset_dt(dt)
        Track._count = 0
        MultiTracker._hist_tracks.clear()

    def init(self, frame, detections):
        """
        Initializes the tracker from detections in the first frame.
        Parameters
        ----------
        frame : ndarray
            Initial frame.
        detections : recarray[DET_DTYPE]
            Record array of N detections.
        """
        self.tracks.clear()
        self.flow.init(frame)
        for det in detections:
            state = self.kf.create(det.tlbr)
            new_trk = Track(0, det.tlbr, state, det.label, self.metric, self.confirm_hits)
            self.tracks[new_trk.trk_id] = new_trk
            LOGGER.debug(f"{'Detected:':<14}{new_trk}")

    def track(self, frame):
        """
        Convenience function that combines `compute_flow` and `apply_kalman`.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        self.compute_flow(frame)
        self.apply_kalman()

    def compute_flow(self, frame):
        """
        Computes optical flow to estimate tracklet positions and camera motion.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        active_tracks = [track for track in self.tracks.values() if track.active]
        self.klt_bboxes, self.homography = self.flow.predict(frame, active_tracks)
        if self.homography is None:
            # clear tracks when camera motion cannot be estimated
            self.tracks.clear()

    def apply_kalman(self):
        """
        Performs kalman filter prediction and update from KLT measurements.
        The function should be called after `compute_flow`.
        """
        for trk_id, track in list(self.tracks.items()):
            mean, cov = track.state
            mean, cov = self.kf.warp(mean, cov, self.homography)
            mean, cov = self.kf.predict(mean, cov)
            if trk_id in self.klt_bboxes:
                klt_tlbr = self.klt_bboxes[trk_id]
                # give large KLT uncertainty for occluded tracks
                # usually these with large age and low inlier ratio
                std_multiplier = max(self.age_penalty * track.age, 1) / track.inlier_ratio
                mean, cov = self.kf.update(mean, cov, klt_tlbr, MeasType.FLOW, std_multiplier)
            next_tlbr = as_rect(mean[:4])
            track.update(next_tlbr, (mean, cov))
            if iom(next_tlbr, self.frame_rect) < 0.5:
                if track.confirmed:
                    LOGGER.info(f"{'Out:':<14}{track}")
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
        occluded_det_ids = self._find_occluded_detections(detections, self.occlusion_thresh)

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

        matches = itertools.chain(matches1, matches2, matches3)
        u_trk_ids = itertools.chain(u_trk_ids1, u_trk_ids2, u_trk_ids3)
        updated, aged = [], []

        # reID with track history
        u_det_ids = {det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh}
        reid_u_det_ids = list(u_det_ids - occluded_det_ids)

        # u_det_ids = [det_id for det_id in u_det_ids if detections[det_id].conf >= self.conf_thresh]

        # reid_u_det_ids = [det_id for det_id in u_det_ids if det_id not in occluded_det_ids
        #                   and detections[det_id].conf >= self.conf_thresh]
        u_detections, u_embeddings = detections[reid_u_det_ids], embeddings[reid_u_det_ids]
        hist_ids = list(MultiTracker._hist_tracks.keys())
        cost = self._reid_cost(u_detections, u_embeddings)
        reid_matches, _, reid_u_det_ids = self._linear_assignment(cost, hist_ids, reid_u_det_ids)

        # reinstate matched tracks
        for trk_id, det_id in reid_matches:
            track = MultiTracker._hist_tracks.pop(trk_id)
            det = detections[det_id]
            LOGGER.info(f"{'Reidentified:':<14}{track}")
            state = self.kf.create(det.tlbr)
            track.reinstate(frame_id, det.tlbr, state, embeddings[det_id])
            self.tracks[trk_id] = track
            updated.append(trk_id)

        # update matched tracks
        for trk_id, det_id in matches:
            track = self.tracks[trk_id]
            det = detections[det_id]
            mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
            next_tlbr = as_rect(mean[:4])
            is_valid = (det_id not in occluded_det_ids)
            track.update(next_tlbr, (mean, cov), embeddings[det_id], is_valid)
            if track.just_confirmed():
                LOGGER.info(f"{'Found:':<14}{track}")
            if iom(next_tlbr, self.frame_rect) < 0.5:
                LOGGER.info(f"{'Out:':<14}{track}")
                self._mark_lost(trk_id)
            else:
                updated.append(trk_id)

        # clean up lost tracks
        for trk_id in u_trk_ids:
            track = self.tracks[trk_id]
            if not track.confirmed:
                LOGGER.debug(f"{'Unconfirmed:':<14}{track}")
                del self.tracks[trk_id]
                continue
            track.mark_missed()
            if track.age > self.max_age:
                LOGGER.info(f"{'Lost:':<14}{track}")
                self._mark_lost(trk_id)
            else:
                aged.append(trk_id)

        u_det_ids = itertools.chain(u_det_ids & occluded_det_ids, reid_u_det_ids)
        # register new detections
        for det_id in u_det_ids:
            det = detections[det_id]
            state = self.kf.create(det.tlbr)
            new_trk = Track(frame_id, det.tlbr, state, det.label, self.metric, self.confirm_hits)
            self.tracks[new_trk.trk_id] = new_trk
            LOGGER.debug(f"{'Detected:':<14}{new_trk}")
            updated.append(new_trk.trk_id)

        # remove duplicate tracks
        self._remove_duplicate(updated, aged)

    def _mark_lost(self, trk_id):
        MultiTracker._hist_tracks[trk_id] = self.tracks.pop(trk_id)
        if len(MultiTracker._hist_tracks) > self.history_size:
            MultiTracker._hist_tracks.popitem(last=False)

    def _matching_cost(self, trk_ids, detections, embeddings):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        # smooth_feats = np.concatenate([self.tracks[trk_id].smooth_feat()
        #                                for trk_id in trk_ids]).reshape(n_trk, -1)
        # cost = cdist(smooth_feats, embeddings, self.metric)

        cost = np.concatenate([self.tracks[trk_id].clust_feat.distance(embeddings)
                               for trk_id in trk_ids]).reshape(n_trk, -1)
        # clust_feats = tuple(self.tracks[trk_id].clust_feat() for trk_id in trk_ids)
        # cost = self._clusters_vec_dist(clust_feats, embeddings, self.metric)

        for row, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            m_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            age = track.age / self.max_age
            self._fuse_motion(cost[row], m_dist, detections.label, age, track.label,
                              self.max_feat_cost, self.motion_weight, self.age_weight)
        return cost

    def _iou_cost(self, trk_ids, detections):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        # make sure associated pair has the same class label
        t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        t_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        d_bboxes = detections.tlbr
        ious = bbox_overlaps(t_bboxes, d_bboxes)
        self._gate_cost(ious, t_labels, detections.label, self.iou_thresh, True)
        return ious

    def _reid_cost(self, detections, embeddings):
        n_hist, n_det = len(MultiTracker._hist_tracks), len(detections)
        if n_hist == 0 or n_det == 0:
            return np.empty((n_hist, n_det))

        t_labels = np.fromiter((t.label for t in MultiTracker._hist_tracks.values()), int, n_hist)
        # smooth_feats = np.concatenate([t.smooth_feat() for t
        #                                in MultiTracker._hist_tracks.values()]).reshape(n_hist, -1)
        # cost = cdist(smooth_feats, embeddings, self.metric)
        cost = np.concatenate([t.clust_feat.distance(embeddings)
                               for t in MultiTracker._hist_tracks.values()]).reshape(n_hist, -1)
        self._gate_cost(cost, t_labels, detections.label, self.max_reid_cost)
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
            LOGGER.debug(f"{'Duplicate:':<14}{self.tracks[trk_id]}")
            del self.tracks[trk_id]

    @staticmethod
    @nb.njit(cache=True)
    def _find_occluded_detections(detections, occlusion_thresh):
        occluded_det_ids = set()
        for i, det in enumerate(detections):
            for j, other in enumerate(detections):
                if i != j and ios(det.tlbr, other.tlbr) > occlusion_thresh:
                    occluded_det_ids.add(i)
                    break
        return occluded_det_ids

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

    # @staticmethod
    # @nb.njit(fastmath=True, cache=True)
    # def _fuse_feature(f_clust_dist, d_labels, out, label, max_feat_cost):
    #     # out[:] = np.minimum(f_smooth_dist, f_clust_dist) * 0.5
    #     # out[:] = f_clust_dist * 0.5
    #     out[:] = f_smooth_dist * 0.5
    #     gate = (out > max_feat_cost) | (label != d_labels)
    #     out[gate] = INF_COST

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fuse_motion(cost, m_dist, d_labels, age, label, max_cost, w1, w2):
        gate = (cost > max_cost) | (m_dist > CHI_SQ_INV_95) | (label != d_labels)
        norm_factor = 1. / CHI_SQ_INV_95
        cost[:] = (1. - w1 - w2) * 0.5 * cost + w1 * norm_factor * m_dist + w2 * age
        cost[gate] = INF_COST

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _gate_cost(cost, t_labels, d_labels, thresh, maximize=False):
        for i in nb.prange(len(cost)):
            if maximize:
                gate = (cost[i] < thresh) | (t_labels[i] != d_labels)
                cost[i][gate] = 0.
            else:
                gate = (cost[i] > thresh) | (t_labels[i] != d_labels)
                cost[i][gate] = INF_COST

    # @staticmethod
    # @nb.njit(parallel=True, fastmath=True, cache=True)
    # def _clusters_vec_dist(clust_feats, embeddings, metric):
    #     clust_feats = list(clust_feats)
    #     dist = np.empty((len(clust_feats), len(embeddings)))
    #     for i in nb.prange(len(clust_feats)):
    #         dist[i, :] = apply_along_axis(np.min, cdist(clust_feats[i], embeddings, metric), axis=0)
    #     return dist
