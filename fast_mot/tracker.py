from pathlib import Path
import itertools
import json

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps
import numpy as np
import numba as nb
import cv2
import time

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils import *


CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of the chi-square distribution with 4 dof
INF_COST = 1e5


class MultiTracker:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['MultiTracker']

    def __init__(self, size, dt, metric): #detector_region):
        self.size = size
        self.metric = metric
        # self.detector_region = detector_region
        self.max_age = MultiTracker.config['max_age']
        self.age_factor = MultiTracker.config['age_factor']
        self.motion_weight = MultiTracker.config['motion_weight']
        self.max_feature_cost = MultiTracker.config['max_feature_cost']
        self.min_iou_cost = MultiTracker.config['min_iou_cost']
        self.dup_iou_thresh = MultiTracker.config['dup_iou_thresh']
        self.max_feat_overlap = MultiTracker.config['max_feat_overlap']
        self.min_register_conf = MultiTracker.config['min_register_conf']
        self.n_init = MultiTracker.config['n_init']
        
        self.next_id = 1
        self.tracks = {}
        self.kf = KalmanFilter(dt, self.n_init)
        self.flow = Flow(self.size, estimate_camera_motion=True)
        # self.frame_rect = Rect(tlwh=(0, 0, *self.size))
        self.frame_rect = to_tlbr((0, 0, *self.size))

        self.flow_bboxes = {}
        self.H_camera = None

    def compute_flow(self, frame):
        self.flow_bboxes, self.H_camera = self.flow.predict(frame, self.tracks)
        if self.H_camera is None:
            # clear tracks when camera motion estimation failed
            self.tracks.clear()

    def step_kalman_filter(self):
        for trk_id, track in list(self.tracks.items()):
            track.frames_since_acquired += 1
            flow_bbox = self.flow_bboxes.get(trk_id)
            if track.frames_since_acquired <= self.n_init:
                if flow_bbox is not None:
                    if track.frames_since_acquired == self.n_init:
                        # initialize kalman filter
                        track.state = self.kf.initiate(track.init_tlbr, flow_bbox)
                    else:
                        track.init_tlbr = warp(track.init_tlbr, self.H_camera) 
                        # track.init_bbox = track.init_tlbr.warp(H_camera)
                        track.tlbr = flow_bbox
                else:
                    print('[Tracker] Lost (init) %s' % track)
                    del self.tracks[trk_id]
            else:
                mean, cov = track.state
                # track using kalman filter and flow measurement
                mean, cov = self.kf.warp(mean, cov, self.H_camera)
                mean, cov = self.kf.predict(mean, cov)
                if flow_bbox is not None and track.active:
                    # std_multiplier = 1
                    # give large flow uncertainty for occluded objects
                    std_multiplier = max(self.age_factor * track.age, 1)
                    mean, cov = self.kf.update(mean, cov, flow_bbox, MeasType.FLOW, std_multiplier)
                # next_bbox = Rect(tlbr=mean[:4])
                # inside_bbox = next_bbox & self.frame_rect
                next_tlbr = as_rect(mean[:4])
                if intersection(next_tlbr, self.frame_rect) is not None:
                    track.state = (mean, cov)
                    track.tlbr = next_tlbr
                else:
                    print('[Tracker] Lost (outside frame) %s' % track)
                    del self.tracks[trk_id]

    def track(self, frame):
        tic = time.perf_counter()
        self.compute_flow(frame)
        print('flow', time.perf_counter() - tic)
        tic = time.perf_counter()
        self.step_kalman_filter()
        print('kalman filter', time.perf_counter() - tic)

    def initiate(self, frame, detections):
        """
        Initialize the tracker from detections in the first frame
        """
        if self.tracks:
            self.tracks.clear()
        self.flow.initiate(frame)
        for det in detections:
            new_track = Track(det.tlbr, det.label, self.next_id)
            self.tracks[self.next_id] = new_track
            print('[Tracker] Found %s' % new_track)
            self.next_id += 1

    def update(self, detections, embeddings):
        """
        Update tracks using detections
        """
        # tic = time.perf_counter()

        det_ids = list(range(len(detections)))
        confirmed = [trk_id for trk_id, track in self.tracks.items() if track.confirmed]
        unconfirmed = [trk_id for trk_id, track in self.tracks.items() if not track.confirmed]

        # association with motion and embeddings
        cost = self._matching_cost(confirmed, detections, embeddings)
        matches_a, u_trk_ids_a, u_det_ids = self._linear_assignment(cost, confirmed, det_ids)

        # 2nd association with iou
        candidates = unconfirmed + [trk_id for trk_id in u_trk_ids_a if self.tracks[trk_id].active]
        u_trk_ids_a = [trk_id for trk_id in u_trk_ids_a if not self.tracks[trk_id].active]

        u_detections = detections[u_det_ids]
        # u_detections = [detections[det_id] for det_id in u_det_ids]
        
        ious = self._iou_cost(candidates, u_detections)
        matches_b, u_trk_ids_b, u_det_ids = self._linear_assignment(ious, candidates, u_det_ids, maximize=True)

        matches = matches_a + matches_b
        u_trk_ids = u_trk_ids_a + u_trk_ids_b

        max_overlaps = self._max_overlaps(matches, confirmed, detections)
        active, inactive = [], []

        # update matched tracks
        for (trk_id, det_id), max_overlap in zip(matches, max_overlaps):
        # for trk_id, det_id in matches:
            track = self.tracks[trk_id]
            det = detections[det_id]
            track.age = 0
            # mean, cov = self.kf.update(*track.state, det.bbox, MeasType.DETECTOR)
            # next_bbox = Rect(tlbr=mean[:4])
            # inside_bbox = next_bbox & self.frame_rect
            mean, cov = self.kf.update(*track.state, det.tlbr, MeasType.DETECTOR)
            next_tlbr = as_rect(mean[:4])
            if intersection(next_tlbr, self.frame_rect) is not None:
                track.state = (mean, cov)
                track.tlbr = next_tlbr
                if not track.confirmed or max_overlap <= self.max_feat_overlap:
                    # if trk_id == 1:
                    #     cv2.imwrite(f'test/target_{trk_id}_{det_id}.jpg', det.bbox.crop(frame))
                    track.update_features(embeddings[det_id])
                if not track.confirmed:
                    track.confirmed = True
                active.append(trk_id)
            else:
                print('[Tracker] Lost (outside frame) %s' % track)
                del self.tracks[trk_id]
        # print('MATCHING', time.perf_counter() - tic)

        # clean up lost tracks
        for trk_id in u_trk_ids:
            track = self.tracks[trk_id]
            track.age += 1
            if not track.confirmed or track.age > self.max_age:
                print('[Tracker] Lost (age) %s' % track)
                del self.tracks[trk_id]
            else:
                inactive.append(trk_id)

        # register new detections
        for det_id in u_det_ids:
            det = detections[det_id]
            if det.conf > self.min_register_conf:
                new_track = Track(det.tlbr, det.label, self.next_id)
                self.tracks[self.next_id] = new_track
                print('[Tracker] Found %s' % new_track)
                # if self.next_id == 32:
                #     print('det id for 32:', det_id)
                active.append(self.next_id)
                self.next_id += 1

        self._remove_duplicate(active, inactive)

    def _matching_cost(self, trk_ids, detections, embeddings):
        # cost = np.empty((len(trk_ids), len(detections)))
        # feature_cost = np.empty((len(trk_ids), len(detections)))
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections))) #, np.empty((len(trk_ids), len(detections)))
            # return cost, feature_cost

        # measurements = np.array([det.bbox.tlbr for det in detections])
        # det_labels = np.array([det.label for det in detections])

        cost = self._feature_distance(trk_ids, embeddings)
        # feature_cost = cost.copy()
        for i, trk_id in enumerate(trk_ids):
            track = self.tracks[trk_id]
            # feature_cost[i] = self._feature_distance(trk_id, embeddings)
            # motion_dist = self.kf.motion_distance(*track.state, measurements)
            # print(motion_dist)
            # if trk_id == 7:
                # print('feature:', cost[i])
                # print('motion:', motion_dist)
            # cost[i] = self._fuse_motion(cost[i], motion_dist, self.max_feature_cost, 
            #     track.label, det_labels, self.motion_weight)

            motion_dist = self.kf.motion_distance(*track.state, detections.tlbr)
            cost[i] = self._fuse_motion(cost[i], motion_dist, self.max_feature_cost, 
                track.label, detections.label, self.motion_weight)
            # gate = (cost[i] > self.max_feature_cost) | (motion_dist > self.max_motion_cost) | \
            #     (track.label != det_labels)
            # cost[i] = (1 - self.motion_weight) * cost[i] + self.motion_weight * motion_dist
            # # gate = (feature_cost[i] > self.max_feature_cost) | (motion_dist > self.max_motion_cost) | \
            # #     (track.label != det_labels)
            # # cost[i] = (1 - self.motion_weight) * feature_cost[i] + self.motion_weight * motion_dist
            # cost[i, gate] = INF_COST
        # print(cost)
        return cost #, feature_cost

    def _iou_cost(self, trk_ids, detections):
        if len(trk_ids) == 0 or len(detections) == 0:
            return np.empty((len(trk_ids), len(detections)))

        # make sure associated pair has the same class label
        trk_labels = np.array([self.tracks[trk_id].label for trk_id in trk_ids])
        # det_labels = np.array([det.label for det in detections])
        
        # trk_bboxes = np.ascontiguousarray(
        #     [self.tracks[trk_id].bbox.tlbr for trk_id in trk_ids],
        #     dtype=np.float
        # )
        # det_bboxes = np.ascontiguousarray(
        #     [det.bbox.tlbr for det in detections],
        #     dtype=np.float
        # )

        trk_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])
        det_bboxes = detections.tlbr
        ious = bbox_overlaps(trk_bboxes, det_bboxes)
        # trk_ids = np.asarray(trk_ids)
        # print('iou', ious[trk_ids == 7])
        # print(ious)
        # ious = self._gate_ious(ious, self.min_iou_cost, trk_labels, det_labels)
        ious = self._gate_ious(ious, self.min_iou_cost, trk_labels, detections.label)
        return ious

    def _linear_assignment(self, cost, trk_ids, det_ids, maximize=False, feature_cost=None):
        rows, cols = linear_sum_assignment(cost, maximize)
        unmatched_rows = list(set(range(cost.shape[0])) - set(rows))
        unmatched_cols = list(set(range(cost.shape[1])) - set(cols))
        unmatched_trk_ids = [trk_ids[row] for row in unmatched_rows]
        unmatched_det_ids = [det_ids[col] for col in unmatched_cols]
        matches = []
        if not maximize:
            for row, col in zip(rows, cols):
                if cost[row, col] < INF_COST:
                    # print(f'matched feature_cost: {feature_cost[row][col]}')
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

    def _feature_distance(self, trk_ids, embeddings):
        features = [self.tracks[trk_id].smooth_feature for trk_id in trk_ids]
        feature_dist = cdist(features, embeddings, self.metric)
        # print(feature_dist)
        return feature_dist

    # def _feature_distance(self, trk_id, embeddings):
    #     feature_dist = cdist(self.tracks[trk_id].features, embeddings, self.metric).min(axis=0)
    #     print(feature_dist)
    #     return feature_dist

    def _max_overlaps(self, matches, trk_ids, detections):
        # return np.zeros(len(matches))
        if len(trk_ids) == 0 or len(matches) == 0:
            return np.zeros(len(matches))
            
        # det_bboxes = np.ascontiguousarray(
        #     [detections[det_id].bbox.tlbr for _, det_id in matches],
        #     dtype=np.float
        # )
        # trk_bboxes = np.ascontiguousarray(
        #     [self.tracks[trk_id].bbox.tlbr for trk_id in trk_ids],
        #     dtype=np.float
        # )
        matched_trk_ids, matched_det_ids = zip(*matches)
        det_bboxes = detections[list(matched_det_ids)].tlbr
        # det_bboxes = np.array([detections[det_id].tlbr for _, det_id in matches])
        trk_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in trk_ids])

        ious = bbox_overlaps(det_bboxes, trk_bboxes)
        diff_id = np.asarray(matched_trk_ids).reshape(-1, 1) != trk_ids
        max_overlaps = ious.max(axis=1, initial=0, where=diff_id)
        return max_overlaps

    def _remove_duplicate(self, active, inactive):
        # active = [trk_id for trk_id, track in self.tracks.items() if track.age == 0]
        # lost = [trk_id for trk_id, track in self.tracks.items() if track.age != 0]
        if len(active) == 0 or len(inactive) == 0:
            return

        # active_bboxes = np.ascontiguousarray(
        #     [self.tracks[trk_id].bbox.tlbr for trk_id in active],
        #     dtype=np.float
        # )
        # inactive_bboxes = np.ascontiguousarray(
        #     [self.tracks[trk_id].bbox.tlbr for trk_id in inactive],
        #     dtype=np.float
        # )

        active_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in active])
        inactive_bboxes = np.array([self.tracks[trk_id].tlbr for trk_id in inactive])

        ious = bbox_overlaps(active_bboxes, inactive_bboxes)
        idx = np.where(ious > self.dup_iou_thresh)
        for row, col in zip(*idx):
            active_id, inactive_id = active[row], inactive[col]
            if self.tracks[active_id].frames_since_acquired >= self.tracks[inactive_id].frames_since_acquired:
                print(f'{inactive_id} removed **************************')
                del self.tracks[inactive_id] # bug here
            else:
                print(f'{active_id} removed **************************')
                del self.tracks[active_id]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fuse_motion(cost, motion_dist, max_cost, label, det_labels, weight):
        gate = (cost > max_cost) | (motion_dist > CHI_SQ_INV_95) | (label != det_labels)
        cost = (1 - weight) * cost + weight * motion_dist
        cost[gate] = INF_COST
        return cost

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _gate_ious(ious, min_iou, trk_labels, det_labels):
        for i in nb.prange(len(ious)):
            gate = (ious[i] < min_iou) | (trk_labels[i] != det_labels)
            ious[i][gate] = 0
        return ious