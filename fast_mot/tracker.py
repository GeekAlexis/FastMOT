from enum import Enum
from pathlib import Path
import json

from collections import OrderedDict
# from numba.typed import Dict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps
import numpy as np
import numba as nb
import math
import cv2
import time

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils import * 


CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of the chi-square distribution with 4 degrees of freedom
INF_COST = 1e5


class MultiTracker:

    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['MultiTracker']

    def __init__(self, size, dt):
        self.size = size
        self.max_age = MultiTracker.config['max_age']
        self.motion_weight = MultiTracker.config['motion_weight']
        self.max_motion_cost = MultiTracker.config['max_motion_cost']
        self.max_feature_cost = MultiTracker.config['max_feature_cost']
        # self.max_motion_cost = math.sqrt(MultiTracker.CHI_SQ_INV_95)
        self.min_iou_cost = MultiTracker.config['min_iou_cost']
        self.feature_buf_size = MultiTracker.config['feature_buf_size']
        self.min_register_conf = MultiTracker.config['min_register_conf']
        self.vertical_bin_height = MultiTracker.config['vertical_bin_height']
        self.n_init = MultiTracker.config['n_init']
        
        self.prev_frame_gray = None
        self.prev_frame_small = None
        # self.prev_pyramid = None
        self.next_id = 0
        self.tracks = OrderedDict()
        self.kf = KalmanFilter(dt, self.n_init)
        self.flow = Flow(self.size, estimate_camera_motion=True)

    def step_flow(self, frame):
        tic = time.perf_counter()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.flow.optflow_scaling[0], fy=self.flow.optflow_scaling[1])
        self.tracks = OrderedDict(sorted(self.tracks.items(), key=self._compare_dist, reverse=True))
        print('gray and sort:', time.perf_counter() - tic)

        # tic = time.perf_counter()
        flow_bboxes, H_camera = self.flow.predict(self.tracks, self.prev_frame_gray, self.prev_frame_small, frame_small)
        if H_camera is None:
            # clear tracks when camera motion estimation failed
            self.tracks.clear()

        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small
        # self.prev_pyramid = pyramid
        # print('opt flow:', time.perf_counter() - tic)
        return flow_bboxes, H_camera

    def step_kf(self, flow_meas, H_camera):
        for track_id, track in list(self.tracks.items()):
            track.frames_since_acquired += 1
            if track.frames_since_acquired <= self.n_init:
                if track_id in flow_meas:
                    flow_bbox = flow_meas[track_id]
                    if track.frames_since_acquired == self.n_init:
                        # initialize kalman filter
                        track.state = self.kf.initiate(track.init_bbox, flow_bbox)
                    else:
                        track.init_bbox = track.init_bbox.warp(H_camera)
                        track.bbox = flow_bbox
                else:
                    print('[Tracker] Target lost (init): %s' % track)
                    del self.tracks[track_id]
            else:
                mean, cov = track.state
                # track using kalman filter and flow measurement
                mean, cov = self.kf.warp(mean, cov, H_camera)
                mean, cov = self.kf.predict(mean, cov)
                if track_id in flow_meas:
                    flow_bbox = flow_meas[track_id]
                    conf = 0.5 / track.age if track.age > 0 else 1
                    mean, cov = self.kf.update(mean, cov, flow_bbox.tlbr, MeasType.FLOW, conf)
                # check for out of frame case
                next_bbox = Rect(tlbr=mean[:4])
                inside_bbox = next_bbox & Rect(tlwh=(0, 0, *self.size))
                if inside_bbox is not None:
                    track.state = (mean, cov)
                    track.bbox = next_bbox
                else:
                    print('[Tracker] Target lost (outside frame): %s' % track)
                    del self.tracks[track_id]

    def track(self, frame):
        tic = time.perf_counter()
        flow_meas, H_camera = self.step_flow(frame)
        print('flow', time.perf_counter() - tic)
        tic = time.perf_counter()
        self.step_kf(flow_meas, H_camera)
        print('kalman filter', time.perf_counter() - tic)

    def initiate(self, frame, detections):
        """
        Initialize the tracker from detections in the first frame
        """
        if self.tracks:
            self.tracks.clear()

        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.flow.optflow_scaling[0],
            fy=self.flow.optflow_scaling[1])

        for det in detections:
            new_track = Track(det.label, det.bbox, self.next_id, self.feature_buf_size)
            self.tracks[self.next_id] = new_track
            print('[Tracker] Track registered: %s' % new_track)
            self.next_id += 1

    def update(self, detections, embeddings):
        """
        Update tracks using detections
        """
        # tic = time.perf_counter()
        # compute optimal assignment

        det_ids = list(range(len(detections)))
        track_ids = [track_id for track_id, track in self.tracks.items() if track.confirmed]
        unconfirmed = [track_id for track_id, track in self.tracks.items() if not track.confirmed]

        # association with motion and embeddings
        cost = self._matching_cost(track_ids, detections, embeddings)
        matches_a, u_track_ids_a, u_det_ids = self._linear_assignment(cost, track_ids, det_ids)

        # 2nd association with iou
        candidates = unconfirmed + [track_id for track_id in u_track_ids_a if self.tracks[track_id].age == 0]
        u_track_ids_a = [track_id for track_id in u_track_ids_a if self.tracks[track_id].age != 0]
        u_detections = [detections[det_id] for det_id in u_det_ids]
        
        ious = self._iou_cost(candidates, u_detections)
        matches_b, u_track_ids_b, u_det_ids = self._linear_assignment(ious, candidates, u_det_ids, maximize=True)

        matches = matches_a + matches_b
        u_track_ids = u_track_ids_a + u_track_ids_b

        # update matched tracks
        for track_id, det_id in matches:
            track = self.tracks[track_id]
            det = detections[det_id]
            track.age = 0
            mean, cov = self.kf.update(*track.state, det.bbox.tlbr, MeasType.DETECTOR)
            next_bbox = Rect(tlbr=mean[:4])
            inside_bbox = next_bbox & Rect(tlwh=(0, 0, *self.size))
            if inside_bbox is not None:
                track.state = (mean, cov)
                track.bbox = next_bbox
                track.update_features(embeddings[det_id])
                track.confirmed = True
            else:
                print('[Tracker] Target lost (out of frame): %s' % track)
                del self.tracks[track_id]
        # print('MATCHING', time.perf_counter() - tic)
    
        # register new detections
        for det_id in u_det_ids:
            det = detections[det_id]
            if det.conf > self.min_register_conf:
                new_track = Track(det.label, det.bbox, self.next_id, self.feature_buf_size)
                self.tracks[self.next_id] = new_track
                print('[Tracker] Track registered: %s' % new_track)
                self.next_id += 1

        # clean up lost tracks
        for track_id in u_track_ids:
            track = self.tracks[track_id]
            if not track.confirmed:
                print('[Tracker] Target lost (unconfirmed): %s' % track)
                del self.tracks[track_id]
                continue
            track.age += 1
            if track.age > self.max_age:
                print('[Tracker] Target lost (age): %s' % track)
                del self.tracks[track_id]

    def _compare_dist(self, id_track_pair):
        # estimate distance using bottow right y coord and area
        return (math.ceil(id_track_pair[1].bbox.ymax / self.vertical_bin_height), id_track_pair[1].bbox.area)

    def _matching_cost(self, track_ids, detections, embeddings):
        if len(track_ids) == 0 or len(detections) == 0:
            return np.empty((len(track_ids), len(detections)))

        measurements = np.array([det.bbox.tlbr for det in detections])
        det_labels = np.array([det.label for det in detections])

        cost = self._feature_distance(track_ids, embeddings)
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            motion_dist = self.kf.motion_distance(*track.state, measurements)
            gate = (cost[i, :] > self.max_feature_cost) | (motion_dist > self.max_motion_cost) | \
                (track.label != det_labels)
            cost[i, :] = (1 - self.motion_weight) * cost[i, :] + self.motion_weight * motion_dist
            cost[i, gate] = INF_COST
        return cost

    def _iou_cost(self, track_ids, detections):
        if len(track_ids) == 0 or len(detections) == 0:
            return np.empty((len(track_ids), len(detections)))

        # make sure associated pair has the same class label
        track_labels = np.array([self.tracks[track_id].label for track_id in track_ids])
        det_labels = np.array([det.label for det in detections])
        diff_labels = track_labels.reshape(-1, 1) != det_labels
        
        track_bboxes = np.ascontiguousarray(
            [self.tracks[track_id].bbox.tlbr for track_id in track_ids],
            dtype=np.float
        )
        det_bboxes = np.ascontiguousarray(
            [det.bbox.tlbr for det in detections],
            dtype=np.float
        )
        ious = bbox_overlaps(track_bboxes, det_bboxes)
        # print(ious)
        gate = (ious < self.min_iou_cost) | diff_labels
        ious[gate] = 0
        return ious

    def _linear_assignment(self, cost, track_ids, det_ids, maximize=False):
        rows, cols = linear_sum_assignment(cost, maximize)
        unmatched_rows = list(set(range(cost.shape[0])) - set(rows))
        unmatched_cols = list(set(range(cost.shape[1])) - set(cols))
        unmatched_track_ids = [track_ids[row] for row in unmatched_rows]
        unmatched_det_ids = [det_ids[col] for col in unmatched_cols]
        matches = []
        if not maximize:
            for row, col in zip(rows, cols):
                if cost[row, col] < INF_COST:
                    matches.append((track_ids[row], det_ids[col]))
                else:
                    unmatched_track_ids.append(track_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        else:
            for row, col in zip(rows, cols):
                if cost[row, col] > 0:
                    matches.append((track_ids[row], det_ids[col]))
                else:
                    unmatched_track_ids.append(track_ids[row])
                    unmatched_det_ids.append(det_ids[col])
        return matches, unmatched_track_ids, unmatched_det_ids

    # @staticmethod
    # @nb.njit(fastmath=True, parallel=True)
    # def _gate_cost_matrix(track_labels, det_labels, motion_cost, appearance_cost, 
    #     max_motion_cost, max_appearance_cost, weight, inf_cost):
    #     diff_label_mask = (track_labels.reshape(-1, 1) != det_labels)
    #     gate_mask = (diff_label_mask | (motion_cost > max_motion_cost) | (appearance_cost > max_appearance_cost))
    #     cost = weight * motion_cost + (1 - weight) * appearance_cost
    #     cost[gate_mask] = inf_cost
    #     return cost

    def _feature_distance(self, track_ids, embeddings):
        features = [self.tracks[track_id].smooth_feature for track_id in track_ids]
        return cdist(features, embeddings, 'cosine')
        # return cdist(track.features, embeddings, 'cosine').min(axis=0)