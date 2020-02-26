import math
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular
import numpy as np
import cv2
import flowtracker
from util import *


class MeasType:
    """
    Enumeration type for Kalman Filter measurement types
    """
    FLOW = 0
    CNN = 1


class BBoxTracker:
    # 0.95 quantile of the chi-square distribution with 4 degrees of freedom
    CHI_SQ_INV_95 = 9.4877
    INF_COST = 1e5

    def __init__(self, size, dt):
        self.size = size
        self.dt = dt
        self.acquisition_max_age = 10
        self.tracking_max_age = 3
        self.acquire = True
        self.max_association_maha = math.sqrt(BBoxTracker.CHI_SQ_INV_95)
        self.min_association_iou = 0.2
        self.min_register_conf = 0.6
        self.num_vertical_bin = 36
        self.n_init = 3
        
        self.small_size_std_acc = (32, 500)
        self.large_size_std_acc = (256, 3000) # max(w, h)
        self.acc_cov = np.diag(np.array([0.25 * self.dt**4] * 4 + [self.dt**2] * 4, dtype=np.float32))
        self.acc_cov[4:, :4] = np.eye(4, dtype=np.float32) * (0.5 * self.dt**3)
        self.acc_cov[:4, 4:] = np.eye(4, dtype=np.float32) * (0.5 * self.dt**3)
        self.meas_mat = np.eye(4, 8, dtype=np.float32)

        self.min_std_cnn = (5, 5)
        self.min_std_flow = (5, 5)
        self.std_factor_cnn = (0.16, 0.16)
        self.std_factor_flow = (0.16, 0.16)
        self.init_std_pos_factor = 20
        self.init_std_vel_factor = 10
        self.vel_coupling = 0.6
        self.vel_half_life = 1
        self.max_vel = 10000
        self.min_size = 25

        self.prev_frame_gray = None
        self.prev_frame_small = None
        # self.prev_pyramid = None
        self.tracks = OrderedDict()
        self.kalman_filters = {}
        self.flow_tracker = flowtracker.FlowTracker(self.size, estimate_camera_motion=True)

    def track(self, frame, use_flow=True):
        """
        Track targets across frames. This function should be called in every frame.
        """
        assert self.prev_frame_gray is not None
        assert self.prev_frame_small is not None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.flow_tracker.optflow_scaling[0], fy=self.flow_tracker.optflow_scaling[1])
        self.tracks = OrderedDict(sorted(self.tracks.items(), key=self._compare_dist, reverse=True))
        flow_tracks = deepcopy(self.tracks)
        H_camera = self.flow_tracker.predict(flow_tracks, self.prev_frame_gray, self.prev_frame_small, frame_small)
        if H_camera is not None:
            for track_id, track in list(self.tracks.items()):
                track.frames_since_acquired += 1
                if track.frames_since_acquired <= self.n_init:
                    if track_id in flow_tracks:
                        flow_track = flow_tracks[track_id]
                        if track.frames_since_acquired == self.n_init:
                            # initialize kalman filter
                            self.kalman_filters[track_id] = self._create_kalman_filter(track.init_bbox, flow_track.bbox)
                        else:
                            track.init_bbox = self._warp_bbox(track.init_bbox, H_camera)
                            track.bbox = flow_track.bbox
                            track.feature_pts = flow_track.feature_pts
                            track.prev_feature_pts = flow_track.prev_feature_pts
                    else:
                        print('[Tracker] Target lost (flow): %s' % track)
                        del self.tracks[track_id]
                else:
                    # track using kalman filter and flow measurement
                    self._warp_kalman_filters(track_id, H_camera)
                    next_state = self.kalman_filters[track_id].predict()
                    self._clip_vel_and_size(track_id)
                    if use_flow and track_id in flow_tracks:
                        flow_track = flow_tracks[track_id]
                        self.kalman_filters[track_id].measurementNoiseCov = self._compute_meas_cov(flow_track.bbox, MeasType.FLOW, flow_track.conf)
                        flow_meas = self._convert_bbox_to_meas(flow_track.bbox)
                        next_state = self.kalman_filters[track_id].correct(flow_meas)
                        self._clip_vel_and_size(track_id)
                        track.feature_pts = flow_track.feature_pts
                        track.prev_feature_pts = flow_track.prev_feature_pts
                    else:
                        track.feature_pts = None

                    # check for out of frame case
                    next_bbox = self._convert_state_to_bbox(next_state)
                    inside_bbox = next_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
                    if inside_bbox is not None:
                        track.bbox = next_bbox
                        self.kalman_filters[track_id].processNoiseCov = self._compute_acc_cov(next_bbox)
                    else:
                        print('[Tracker] Target lost (outside frame): %s' % track)
                        del self.tracks[track_id]
                        del self.kalman_filters[track_id]
        else:
            # clear tracks when background registration failed
            self.tracks.clear()
            self.kalman_filters.clear()
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small
        # self.prev_pyramid = pyramid

    def init(self, frame, detections):
        """
        Initialize the tracker from detections in the first frame
        """
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.flow_tracker.optflow_scaling[0], fy=self.flow_tracker.optflow_scaling[1])
        for new_track_id, det in enumerate(detections):
            self.tracks[new_track_id] = Track(det.label, det.bbox, new_track_id)
            print('[Tracker] Track registered: %s' % self.tracks[new_track_id])

    def update(self, detections, tile, overlap, acquire=True):
        """
        Update tracks using detections
        """
        # filter out tracks and detections not in tile
        sx = sy = 1 - overlap
        scaled_tile = tile.scale(sx, sy)
        tracks, track_ids, boundary_tracks = ([] for i in range(3))
        use_maha_cost = True
        for track_id, track in self.tracks.items():
            if self.acquire != acquire:
                # reset age when mode toggles
                track.age = 0
            track.age += 1
            if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox): 
                if track_id not in self.kalman_filters:
                    use_maha_cost = False
                track_ids.append(track_id)
                tracks.append(track)
            elif iou(track.bbox, tile) > 0:
                boundary_tracks.append(track)
        self.acquire = acquire

        # compute optimal assignment
        all_det_indices = list(range(len(detections)))
        unmatched_det_indices = all_det_indices
        if len(detections) > 0 and len(tracks) > 0:
            iou_mat = np.array([[iou(track.bbox, det.bbox) for det in detections] for track in tracks], dtype=np.float32)
            if use_maha_cost:
                maha_mat = np.array([[self._maha_dist(track_id, det) for det in detections] for track_id in track_ids], dtype=np.float32)
                # max_maha_dist = np.max(maha_mat)
            else:
                maha_mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
            # print(maha_mat)
                # max_maha_dist = 1
            # print('maha', maha_mat)
            cost = maha_mat + (1 - iou_mat)
            # validation gating using Chi square test and IOU, and ensure same label
            diff_label_mask = np.array([[track.label != det.label for det in detections] for track in tracks])
            cost[diff_label_mask | (iou_mat < self.min_association_iou) | (maha_mat > self.max_association_maha)] = BBoxTracker.INF_COST
            # print('cost', cost)

            track_indices, det_indices = linear_sum_assignment(cost)
            unmatched_det_indices = list(set(all_det_indices) - set(det_indices))
            for track_idx, det_idx in zip(track_indices, det_indices):
                track_id = track_ids[track_idx]
                assert(cost[track_idx, det_idx] <= BBoxTracker.INF_COST)
                if cost[track_idx, det_idx] < BBoxTracker.INF_COST:
                    if track_id in self.kalman_filters:
                        self.kalman_filters[track_id].measurementNoiseCov = self._compute_meas_cov(detections[det_idx].bbox, MeasType.CNN)
                        det_meas = self._convert_bbox_to_meas(detections[det_idx].bbox)
                        next_state = self.kalman_filters[track_id].correct(det_meas)
                        self._clip_vel_and_size(track_id)
                        next_bbox = self._convert_state_to_bbox(next_state)
                        inside_bbox = next_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
                        if inside_bbox is not None:
                            self.tracks[track_id].bbox = next_bbox
                            self.tracks[track_id].age = 0
                            self.kalman_filters[track_id].processNoiseCov = self._compute_acc_cov(next_bbox)
                        else:
                            print('[Tracker] Target lost (out of frame): %s' % self.tracks[track_id])
                            del self.tracks[track_id]
                            del self.kalman_filters[track_id]
                    else:
                        self.tracks[track_id].bbox = detections[det_idx].bbox
                        self.tracks[track_id].age = 0
                else:
                    unmatched_det_indices.append(det_idx)

        # register new detections
        for det_idx in unmatched_det_indices:
            if detections[det_idx].conf > self.min_register_conf:
                register = True
                for track in boundary_tracks:
                    if detections[det_idx].label == track.label and iou(detections[det_idx].bbox, track.bbox) > 0.1:
                        register = False
                if register:
                    new_track_id = 0
                    while new_track_id in self.tracks:
                        new_track_id += 1
                    self.tracks[new_track_id] = Track(detections[det_idx].label, detections[det_idx].bbox, new_track_id)
                    print('[Tracker] Track registered: %s' % self.tracks[new_track_id])

        # clean up lost tracks
        max_age = self.acquisition_max_age if acquire else self.tracking_max_age
        for track_id, track in list(self.tracks.items()):
            if track.age > max_age:
                print('[Tracker] Target lost (age): %s' % self.tracks[track_id])
                del self.tracks[track_id]
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]

    def get_nearest_track(self):
        """
        Compute the nearest track by estimating the relative distance
        """
        if not self.tracks:
            return -1
        nearest_track_id = max(self.tracks.items(), key=self._compare_dist)[0]
        return nearest_track_id

    def _compare_dist(self, id_track_pair):
        # estimate distance using bottow right y coord and area
        bin_height = self.size[1] // self.num_vertical_bin
        return (math.ceil(id_track_pair[1].bbox.ymax / bin_height), id_track_pair[1].bbox.area())

    def _create_kalman_filter(self, init_bbox, cur_bbox):
        kalman_filter = cv2.KalmanFilter(8, 4)
        # constant velocity model
        kalman_filter.transitionMatrix = np.array(
            [[1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt, 0],
             [0, 1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt], 
             [0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt, 0], 
             [0, 0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt], 
             [0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0, 0], 
             [0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0], 
             [0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0],
             [0, 0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life)]], 
            dtype=np.float32
        )
        
        # kalman_filter.transitionMatrix = np.array(
        #     [[1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt, 0],
        #      [0, 1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt], 
        #      [0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt, 0], 
        #      [0, 0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt], 
        #      [0, 0, 0, 0, 1, 0, 0, 0], 
        #      [0, 0, 0, 0, 0, 1, 0, 0], 
        #      [0, 0, 0, 0, 0, 0, 1, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 1]], 
        #     dtype=np.float32
        # )

        kalman_filter.processNoiseCov = self._compute_acc_cov(cur_bbox)
        kalman_filter.measurementMatrix = np.empty_like(self.meas_mat, dtype=np.float32)
        np.copyto(kalman_filter.measurementMatrix, self.meas_mat)
        
        # vels = (np.asarray(cur_bbox.tf_rect()) - np.asarray(init_bbox.tf_rect())) / (self.dt * (self.n_init))
        center_vel = (np.asarray(cur_bbox.center()) - np.asarray(init_bbox.center())) / (self.dt * self.n_init)
        kalman_filter.statePre = np.zeros((8, 1), dtype=np.float32)
        kalman_filter.statePre[:4, 0] = cur_bbox.tf_rect()
        # kalman_filter.statePre[4:, 0] = vels
        kalman_filter.statePre[4:6, 0] = center_vel
        kalman_filter.statePre[6:, 0] = center_vel
        kalman_filter.statePost = np.empty_like(kalman_filter.statePre, dtype=np.float32)
        np.copyto(kalman_filter.statePost, kalman_filter.statePre)

        width, height = cur_bbox.size
        std = np.array([
            self.init_std_pos_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_pos_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_pos_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_pos_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_vel_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_vel_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_vel_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_vel_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            ],
            dtype=np.float32
        )
        kalman_filter.errorCovPost = np.diag(np.square(std))
        return kalman_filter
        
    def _convert_bbox_to_meas(self, bbox):
        # return np.float32(list(bbox.center()) + list(bbox.size)).reshape(4, 1)
        return np.float32(bbox.tf_rect()).reshape(4, 1)

    def _convert_state_to_bbox(self, state):
        return Rect(tf_rect=np.int_(np.round(state[:4, 0])))

    def _compute_meas_cov(self, bbox, meas_type=MeasType.FLOW, conf=1.0):
        width, height = bbox.size
        if meas_type == MeasType.FLOW:
            std_factor = self.std_factor_flow
            min_std = self.min_std_flow
            # print(conf)
        elif meas_type == MeasType.CNN:
            std_factor = self.std_factor_cnn
            min_std = self.min_std_cnn
        std = np.array([
            max(width * std_factor[0], min_std[0]),
            max(height * std_factor[1], min_std[1]),
            max(width * std_factor[0], min_std[0]),
            max(height * std_factor[1], min_std[1])
            ],
            dtype=np.float32
        )
        # conf = 1
        return np.diag(np.square(std / conf)) # TODO: better conf

    def _compute_acc_cov(self, bbox):
        std_acc_growth_rate = (self.large_size_std_acc[1] - self.small_size_std_acc[1]) / (self.large_size_std_acc[0] - self.small_size_std_acc[0])
        std_acc = self.small_size_std_acc[1] + (max(bbox.size) - self.small_size_std_acc[0]) * std_acc_growth_rate
        return self.acc_cov * std_acc**2

    def _clip_vel_and_size(self, track_id):
        kalman_filter = self.kalman_filters[track_id]
        kalman_filter.statePost[4:, 0] = np.clip(kalman_filter.statePost[4:, 0], -self.max_vel, self.max_vel)
        bbox = self._convert_state_to_bbox(kalman_filter.statePost)
        sx = self.min_size / bbox.size[0] if bbox.size[0] < self.min_size else 1
        sy = self.min_size / bbox.size[1] if bbox.size[1] < self.min_size else 1
        kalman_filter.statePost[:4, 0] = bbox.scale(sx, sy).tf_rect()

    def _maha_dist(self, track_id, det):
        kalman_filter = self.kalman_filters[track_id]
        # project state to measurement space
        projected_mean = self.meas_mat @ kalman_filter.statePost
        projected_cov = np.linalg.multi_dot([self.meas_mat, kalman_filter.errorCovPost, self.meas_mat.T])

        # compute innovation and innovation covariance
        meas = self._convert_bbox_to_meas(det.bbox)
        meas_cov = self._compute_meas_cov(det.bbox, MeasType.CNN)
        innovation = meas - projected_mean
        innovation_cov = projected_cov + meas_cov

        # mahalanobis distance
        L = np.linalg.cholesky(innovation_cov)
        x = solve_triangular(L, innovation, lower=True, overwrite_b=True, check_finite=False)
        return math.sqrt(np.sum(x**2))

    def _warp_bbox(self, bbox, H_camera):
        warped_corners = cv2.perspectiveTransform(np.float32(bbox.corners()).reshape(4, 1, 2), H_camera)
        return Rect(cv_rect=cv2.boundingRect(warped_corners))

    def _warp_kalman_filters(self, track_id, H_camera):
        # TODO: translation only?
        kalman_filter = self.kalman_filters[track_id]
        pos_tl = kalman_filter.statePost[:2]
        pos_br = kalman_filter.statePost[2:4]
        vel_tl = kalman_filter.statePost[4:6]
        vel_br = kalman_filter.statePost[6:]
        A = H_camera[:2, :2]
        v = H_camera[2, :2].reshape(1, 2)
        t = H_camera[:2, 2].reshape(2, 1)
        # h33 = H_camera[-1, -1]
        temp = (v @ pos_tl + 1)
        grad_tl = (temp * A - (A @ pos_tl + t) @ v) / temp**2
        temp = (v @ pos_br + 1)
        grad_br = (temp * A - (A @ pos_br + t) @ v) / temp**2

        # vel_tl += pos_tl
        # vel_br += pos_br
        # warped = cv2.perspectiveTransform(np.array([pos_tl.T, pos_br.T, vel_tl.T, vel_br.T]), H_camera)
        # kalman_filter.statePost[:2] = warped[0].T
        # kalman_filter.statePost[2:4] = warped[1].T
        # kalman_filter.statePost[4:6] = warped[2].T - warped[0].T
        # kalman_filter.statePost[6:] = warped[3].T - warped[1].T

        # warp state
        warped_pos = cv2.perspectiveTransform(np.array([pos_tl.T, pos_br.T]), H_camera)
        kalman_filter.statePost[:4, 0] = warped_pos.ravel()
        kalman_filter.statePost[4:6] = grad_tl @ vel_tl
        kalman_filter.statePost[6:] = grad_br @ vel_br

        # warp covariance too
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                grad_left = grad_tl if i // 2 % 2 == 0 else grad_br
                grad_right = grad_tl if j // 2 % 2 == 0 else grad_br
                kalman_filter.errorCovPost[i:i + 2, j:j + 2] = grad_left @ kalman_filter.errorCovPost[i:i + 2, j:j + 2] @ grad_right.T

