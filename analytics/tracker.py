from enum import Enum
from pathlib import Path
import json

from collections import OrderedDict
from numba.typed import Dict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import numba as nb
import cv2
import time

from .track import Track
from .flow import Flow
from .kalman_filter import MeasType, KalmanFilter
from .utils import * 


class MultiTracker:
    # 0.95 quantile of the chi-square distribution with 4 degrees of freedom
    CHI_SQ_INV_95 = 9.4877
    INF_COST = 1e5

    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['MultiTracker']

    def __init__(self, size, dt):
        self.size = size
        self.acquisition_max_age = MultiTracker.config['acquisition_max_age']
        self.tracking_max_age = MultiTracker.config['tracking_max_age']
        self.motion_cost_weight = MultiTracker.config['motion_cost_weight']
        self.max_motion_cost = MultiTracker.config['max_motion_cost']
        self.max_appearance_cost = MultiTracker.config['max_appearance_cost']
        # self.max_motion_cost = np.sqrt(MultiTracker.CHI_SQ_INV_95)
        self.min_association_iou = MultiTracker.config['min_association_iou']
        self.feature_buf_size = MultiTracker.config['feature_buf_size']
        self.min_register_conf = MultiTracker.config['min_register_conf']
        self.vertical_bin_height = MultiTracker.config['vertical_bin_height']
        self.n_init = MultiTracker.config['n_init']
        
        self.acquire = True
        self.all_initialized = True
        self.prev_frame_gray = None
        self.prev_frame_small = None
        # self.prev_pyramid = None
        self.next_id = 0
        self.tracks = OrderedDict()
        # self.states = Dict.empty(
        #     key_type=nb.types.int64,
        #     value_type=nb.types.Tuple((nb.float64[::1], nb.float64[:, ::1])),
        # )
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
        self.all_initialized = True
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
                        self.all_initialized = False
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
                    conf = 0.3 / track.age if track.age > 0 else 1
                    mean, cov = self.kf.update(mean, cov, flow_bbox.tlbr, MeasType.FLOW, conf)
                track.state = (mean, cov)
                # check for out of frame case
                next_bbox = Rect(tlbr=mean[:4])
                inside_bbox = next_bbox & Rect(tlwh=(0, 0, *self.size))
                if inside_bbox is not None:
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

    def initiate(self, frame, detections, embeddings=None):
        """
        Initialize the tracker from detections in the first frame
        """
        if self.tracks:
            self.tracks.clear()
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.flow.optflow_scaling[0],
            fy=self.flow.optflow_scaling[1])
        for i, det in enumerate(detections):
            new_track = Track(det.label, det.bbox, self.next_id, self.feature_buf_size)
            if embeddings is not None:
                new_track.features.append(embeddings[i])
            self.tracks[self.next_id] = new_track
            print('[Tracker] Track registered: %s' % new_track)
            self.next_id += 1

    def update(self, detections, embeddings=None, tile=None, overlap=None, acquire=True):
        """
        Update tracks using detections
        """
        if tile is not None:
            assert overlap is not None
            # handle single batch size differently
            sx = sy = 1 - overlap
            scaled_tile = tile * (sx, sy)

        excluded_track_ids = []
        if tile is None:
            track_ids = [*self.tracks]
        else:
            track_ids = []
            for track_id, track in self.tracks.items():
                if self.acquire != acquire:
                    # reset age when mode toggles
                    track.age = -1
                # filter out tracks and detections not in tile
                if track.bbox.center in scaled_tile or tile.contains_rect(track.bbox): 
                    track_ids.append(track_id)
                elif track.bbox.iou(tile) > 0:
                    excluded_track_ids.append(track_id)
        self.acquire = acquire

        tic = time.perf_counter()
        # compute optimal assignment
        all_det_indices = list(range(len(detections)))
        unmatched_det_indices = all_det_indices
        if len(detections) > 0 and len(track_ids) > 0:
            cost = self._compute_cost_matrix(track_ids, detections, embeddings)
            print('BEFORE_MATCHING', time.perf_counter() - tic)
            track_indices, det_indices = linear_sum_assignment(cost)
            unmatched_det_indices = list(set(all_det_indices) - set(det_indices))
            for track_idx, det_idx in zip(track_indices, det_indices):
                track_id = track_ids[track_idx]
                if cost[track_idx, det_idx] < MultiTracker.INF_COST:
                    track = self.tracks[track_id]
                    track.age = -1
                    if track.state is None:
                        track.bbox = detections[det_idx].bbox
                    else:
                        mean, cov = track.state
                        det_meas = detections[det_idx].bbox.tlbr
                        mean, cov = self.kf.update(mean, cov, det_meas, MeasType.CNN)
                        track.state = (mean, cov)
                        next_bbox = Rect(tlbr=mean[:4])
                        inside_bbox = next_bbox & Rect(tlwh=(0, 0, *self.size))
                        if inside_bbox is not None:
                            track.bbox = next_bbox
                            if embeddings is not None:
                                track.features.append(embeddings[det_idx])
                        else:
                            print('[Tracker] Target lost (out of frame): %s' % track)
                            del self.tracks[track_id]
                else:
                    unmatched_det_indices.append(det_idx)
        # print('association', time.perf_counter() - tic)
    
        # register new detections
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            if det.conf > self.min_register_conf:
                register = True
                for track_id in excluded_track_ids:
                    if det.label == track.label and det.bbox.iou(self.tracks[track_id].bbox) > 0.1:
                        register = False
                        break
                if register:
                    new_track = Track(det.label, det.bbox, self.next_id, self.feature_buf_size)
                    if embeddings is not None:
                        new_track.features.append(embeddings[det_idx])
                    self.tracks[self.next_id] = new_track
                    print('[Tracker] Track registered: %s' % new_track)
                    self.next_id += 1

        # clean up lost tracks
        max_age = self.acquisition_max_age if acquire else self.tracking_max_age
        for track_id, track in list(self.tracks.items()):
            track.age += 1
            if track.age > max_age:
                print('[Tracker] Target lost (age): %s' % self.tracks[track_id])
                del self.tracks[track_id]
    
    def get_nearest_track(self, classes=None):
        """
        Compute the nearest track from certain classes by estimating the relative distance
        """
        if classes is None:
            tracks = self.tracks
        else:
            classes = set(classes)
            tracks = {track_id: track for track_id, track in self.tracks.items() if track.label in classes}
        if not tracks:
            return None
        nearest_track_id = max(tracks.items(), key=self._compare_dist)[0]
        return nearest_track_id

    def _compare_dist(self, id_track_pair):
        # estimate distance using bottow right y coord and area
        return (np.ceil(id_track_pair[1].bbox.ymax / self.vertical_bin_height), id_track_pair[1].bbox.area)

    def _compute_cost_matrix(self, track_ids, detections, embeddings=None):
        iou_only = False
        use_motion_cost = False
        motion_cost = np.zeros((len(track_ids), len(detections)))
        appearance_cost = np.zeros((len(track_ids), len(detections)))
        cost = motion_cost # reuse to avoid extra memory allocation
    
        if self.all_initialized:
            # only use motion cost if all tracks are initialized
            use_motion_cost = True
            measurements = np.array([det.bbox.tlbr for det in detections])
        elif embeddings is None:
            iou_only = True
            det_bboxes = np.array([det.bbox.tlbr for det in detections])
        
        # make sure associated pair has the same class label
        track_labels = np.array([self.tracks[track_id].label for track_id in track_ids])
        det_labels = np.array([det.label for det in detections])
        diff_label_mask = (track_labels.reshape(-1, 1) != det_labels)

        if iou_only:
            for i, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id].bbox.tlbr
                cost[i, :] = -iou(track_bbox, det_bboxes)
            gate_mask = (diff_label_mask | (cost > -self.min_association_iou))
        else:
            for i, track_id in enumerate(track_ids):
                if use_motion_cost:
                    motion_cost[i, :] = self._motion_distance(track_id, measurements)
                if embeddings is not None:
                    appearance_cost[i, :] = self._feature_distance(track_id, embeddings)
            gate_mask = (diff_label_mask | (motion_cost > self.max_motion_cost) | (appearance_cost > self.max_appearance_cost))
            cost[:] = ((self.motion_cost_weight) * motion_cost + (1 - self.motion_cost_weight) * appearance_cost)
        # print(appearance_cost)
            
        # gate cost matrix
        cost[gate_mask] = MultiTracker.INF_COST
        return cost

    # @staticmethod
    # @nb.njit(fastmath=True, parallel=True)
    # def _gate_cost_matrix(track_labels, det_labels, motion_cost, appearance_cost, 
    #     max_motion_cost, max_appearance_cost, weight, inf_cost):
    #     diff_label_mask = (track_labels.reshape(-1, 1) != det_labels)
    #     gate_mask = (diff_label_mask | (motion_cost > max_motion_cost) | (appearance_cost > max_appearance_cost))
    #     cost = weight * motion_cost + (1 - weight) * appearance_cost
    #     cost[gate_mask] = inf_cost
    #     return cost

    def _motion_distance(self, track_id, measurements):
        mean, cov = self.tracks[track_id].state
        return self.kf.motion_distance(mean, cov, measurements)

    def _feature_distance(self, track_id, embeddings):
        track = self.tracks[track_id]
        return cdist(track.features, embeddings, 'cosine').min(axis=0)
