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
from .utils.numba import delete_row_col
from .utils.rect import as_rect, to_tlbr, ios, iom


LOGGER = logging.getLogger(__name__)
CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of chi-square distribution
INF_COST = 1e5


class MultiTracker:
    _hist_tracks = OrderedDict()

    def __init__(self, size, metric,
                 max_age=6,
                 age_penalty=2,
                 age_weight=0.05,
                 motion_weight=0.15,
                 max_assoc_cost=0.9,
                 max_reid_cost=0.45,
                 iou_thresh=0.4,
                 duplicate_thresh=0.8,
                 occlusion_thresh=0.7,
                 conf_thresh=0.5,
                 confirm_hits=1,
                 history_size=50,
                 kalman_filter_cfg=None,
                 flow_cfg=None):
        """Class that uses KLT and Kalman filter to track multiple objects and
        associates detections to tracklets based on motion and appearance.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        metric : {'euclidean', 'cosine'}
            Feature distance metric to associate tracklets.
        max_age : int, optional
            Max number of undetected frames allowed before a track is terminated.
        age_penalty : int, optional
            Scale factor to penalize KLT measurements for track with large age.
        age_weight : float, optional
            Weight for tracking age term in matching cost function.
        motion_weight : float, optional
            Weight for motion term in matching cost function.
        max_assoc_cost : float, optional
            Max matching cost for valid primary association.
        max_reid_cost : float, optional
            Max ReID feature dissimilarity for valid reidentification.
        iou_thresh : float, optional
            IoU threshold for valid secondary association
        duplicate_thresh : float, optional
            Track overlap threshold for removing duplicate tracks.
        occlusion_thresh : float, optional
            Detection overlap threshold for nullifying the extracted features for association/reID.
        conf_thresh : float, optional
            Detection confidence threshold for starting a new track.
        confirm_hits : int, optional
            Min number of detections to confirm a track.
        history_size : int, optional
            Max size of track history to keep for reID.
        kalman_filter_cfg : SimpleNamespace, optional
            Kalman Filter configuration.
        flow_cfg : SimpleNamespace, optional
            Flow configuration.
        """
        self.size = size
        self.metric = metric
        assert max_age >= 1
        self.max_age = max_age
        assert age_penalty >= 1
        self.age_penalty = age_penalty
        assert 0 <= age_weight <= 1
        self.age_weight = age_weight
        assert 0 <= motion_weight <= 1
        self.motion_weight = motion_weight
        assert 0 <= max_assoc_cost <= 2
        self.max_assoc_cost = max_assoc_cost
        assert 0 <= max_reid_cost <= 2
        self.max_reid_cost = max_reid_cost
        assert 0 <= iou_thresh <= 1
        self.iou_thresh = iou_thresh
        assert 0 <= duplicate_thresh <= 1
        self.duplicate_thresh = duplicate_thresh
        assert 0 <= occlusion_thresh <= 1
        self.occlusion_thresh = occlusion_thresh
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert confirm_hits >= 1
        self.confirm_hits = confirm_hits
        assert history_size >= 0
        self.history_size = history_size

        self.tracks = {}
        self.kf = KalmanFilter(**vars(kalman_filter_cfg))
        self.flow = Flow(self.size, **vars(flow_cfg))
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.klt_bboxes = {}
        self.homography = None

    def reset(self, dt):
        """Reset the tracker for new input context.

        Parameters
        ----------
        dt : float
            Time interval in seconds between each frame.
        """
        self.kf.reset_dt(dt)
        Track._count = 0
        MultiTracker._hist_tracks.clear()

    def init(self, frame, detections):
        """Initializes the tracker from detections in the first frame.

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
        """Convenience function that combines `compute_flow` and `apply_kalman`.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        self.compute_flow(frame)
        self.apply_kalman()

    def compute_flow(self, frame):
        """Computes optical flow to estimate tracklet positions and camera motion.

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
        """Performs kalman filter predict and update from KLT measurements.
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
            track.update_location(next_tlbr, (mean, cov))
            if iom(next_tlbr, self.frame_rect) < 0.5:
                if track.confirmed:
                    LOGGER.info(f"{'Out:':<14}{track}")
                    self._mark_lost(trk_id)
                else:
                    del self.tracks[trk_id]

    def update(self, frame_id, detections, embeddings):
        """Associates detections to tracklets based on motion and feature embeddings.

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
        cost = self._matching_cost(confirmed, detections, embeddings, occluded_det_ids)
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
        reid_u_det_ids = u_det_ids - occluded_det_ids
        reid_u_det_ids = np.fromiter(reid_u_det_ids, int, len(reid_u_det_ids))

        u_detections, u_embeddings = detections[reid_u_det_ids], embeddings[reid_u_det_ids]
        # hist_ids = list(MultiTracker._hist_tracks.keys())
        cost = self._reid_cost(u_detections, u_embeddings)
        # reid_matches, _, reid_u_det_ids = self._linear_assignment(cost, hist_ids, reid_u_det_ids)

        hist_ids = np.fromiter(MultiTracker._hist_tracks.keys(), int,
                               len(MultiTracker._hist_tracks))
        # reid_u_det_ids = np.asarray(reid_u_det_ids)
        reid_matches, _, reid_u_det_ids = self._greedy_match(cost, hist_ids, reid_u_det_ids,
                                                             self.max_reid_cost)

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
            track.add_detection(frame_id, next_tlbr, (mean, cov), embeddings[det_id], is_valid)
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

    def _matching_cost(self, trk_ids, detections, embeddings, occluded_det_ids):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

        features = np.empty((n_trk, embeddings.shape[1]))
        invalid_rows = []
        for row, trk_id in enumerate(trk_ids):
            avg_feat = self.tracks[trk_id].avg_feat()
            if avg_feat is None:
                invalid_rows.append(row)
            else:
                features[row] = avg_feat
        invalid_rows = np.asarray(invalid_rows, dtype=int)

        # features = np.concatenate([self.tracks[trk_id].avg_feat()
        #                            for trk_id in trk_ids]).reshape(n_trk, -1)
        cost = cdist(features, embeddings, self.metric)

        # cost = np.concatenate([self.tracks[trk_id].clust_feat.distance(embeddings)
        #                        for trk_id in trk_ids]).reshape(n_trk, -1)
        # clust_feats = tuple(self.tracks[trk_id].clust_feat() for trk_id in trk_ids)
        # cost = self._clusters_vec_dist(clust_feats, embeddings, self.metric)

        invalid_cols = np.fromiter(occluded_det_ids, int, len(occluded_det_ids))
        self._invalidate_cost(cost, invalid_rows, invalid_cols)

        for row, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            m_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            age = track.age / self.max_age
            self._fuse_motion(cost[row], m_dist, age, self.motion_weight, self.age_weight)

        # make sure associated pair has the same class label
        t_labels = np.fromiter((self.tracks[trk_id].label for trk_id in trk_ids), int, n_trk)
        self._gate_cost(cost, t_labels, detections.label, self.max_assoc_cost)
        return cost

    def _iou_cost(self, trk_ids, detections):
        n_trk, n_det = len(trk_ids), len(detections)
        if n_trk == 0 or n_det == 0:
            return np.empty((n_trk, n_det))

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

        features = np.empty((n_hist, embeddings.shape[1]))
        invalid_rows = []
        for row, track in enumerate(MultiTracker._hist_tracks.values()):
            avg_feat = track.avg_feat()
            if avg_feat is None:
                invalid_rows.append(row)
            else:
                features[row] = avg_feat
        invalid_rows = np.asarray(invalid_rows, dtype=int)

        # features = np.concatenate([t.avg_feat() for t
        #                            in MultiTracker._hist_tracks.values()]).reshape(n_hist, -1)
        cost = cdist(features, embeddings, self.metric)
        self._invalidate_cost(cost, invalid_rows)

        # cost = np.concatenate([t.clust_feat.distance(embeddings)
        #                        for t in MultiTracker._hist_tracks.values()]).reshape(n_hist, -1)

        t_labels = np.fromiter((t.label for t in MultiTracker._hist_tracks.values()), int, n_hist)
        self._gate_cost(cost, t_labels, detections.label, self.max_reid_cost)
        return cost

    def _remove_duplicate(self, trk_ids_a, trk_ids_b):
        if len(trk_ids_a) == 0 or len(trk_ids_b) == 0:
            return

        bboxes_a = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids_a])
        bboxes_b = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids_b])

        ious = bbox_overlaps(bboxes_a, bboxes_b)
        idx = np.where(ious >= self.duplicate_thresh)
        dup_ids = set()
        for row, col in zip(*idx):
            trk_id_a, trk_id_b = trk_ids_a[row], trk_ids_b[col]
            track_a, track_b = self.tracks[trk_id_a], self.tracks[trk_id_b]
            time_a = (track_a.end_frame - track_a.start_frame, -track_a.start_frame)
            time_b = (track_b.end_frame - track_b.start_frame, -track_b.start_frame)
            if time_a > time_b:
                # track_a.merge_continuation(track_b)
                dup_ids.add(trk_id_b)
            else:
                # track_b.merge_continuation(track_a)
                dup_ids.add(trk_id_a)
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
    def _linear_assignment(cost, row_ids, col_ids, maximize=False):
        rows, cols = linear_sum_assignment(cost, maximize)
        unmatched_rows = list(set(range(cost.shape[0])) - set(rows))
        unmatched_cols = list(set(range(cost.shape[1])) - set(cols))
        unmatched_row_ids = [row_ids[row] for row in unmatched_rows]
        unmatched_col_ids = [col_ids[col] for col in unmatched_cols]
        matches = []
        if not maximize:
            for row, col in zip(rows, cols):
                if cost[row, col] < INF_COST:
                    matches.append((row_ids[row], col_ids[col]))
                else:
                    unmatched_row_ids.append(row_ids[row])
                    unmatched_col_ids.append(col_ids[col])
        else:
            for row, col in zip(rows, cols):
                if cost[row, col] > 0:
                    matches.append((row_ids[row], col_ids[col]))
                else:
                    unmatched_row_ids.append(row_ids[row])
                    unmatched_col_ids.append(col_ids[col])
        return matches, unmatched_row_ids, unmatched_col_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _greedy_match(cost, row_ids, col_ids, thresh):
        indices_rows = np.arange(cost.shape[0])
        indices_cols = np.arange(cost.shape[1])

        matches = []
        while len(indices_rows) > 0 and len(indices_cols) > 0:
            idx = np.argmin(cost)
            i, j = idx // cost.shape[1], idx % cost.shape[1]
            if cost[i, j] <= thresh:
                matches.append((row_ids[indices_rows[i]], col_ids[indices_cols[j]]))
                indices_rows = np.delete(indices_rows, i)
                indices_cols = np.delete(indices_cols, j)
                cost = delete_row_col(cost, i, j)
            else:
                break

        unmatched_row_ids = [row_ids[row] for row in indices_rows]
        unmatched_col_ids = [col_ids[col] for col in indices_cols]
        return matches, unmatched_row_ids, unmatched_col_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fuse_motion(cost, m_dist, age, w1, w2):
        norm_factor = 1. / CHI_SQ_INV_95
        cost[:] = (1. - w1 - w2) * cost + w1 * norm_factor * m_dist + w2 * age
        cost[m_dist > CHI_SQ_INV_95] = INF_COST

    @staticmethod
    @nb.njit(cache=True)
    def _invalidate_cost(cost, invalid_rows=None, invalid_cols=None, val=1.0):
        if invalid_rows is not None:
            cost[invalid_rows, :] = val
        if invalid_cols is not None:
            cost[:, invalid_cols] = val

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
