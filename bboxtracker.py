import math
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
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
    def __init__(self, size, dt):
        self.size = size
        self.dt = dt
        self.acquisition_max_age = 10
        self.tracking_max_age = 3
        self.acquire = True
        self.min_association_iou = 0.25
        self.min_register_conf = 0.6
        self.num_vertical_bin = 20
        self.n_init = 8
        
        self.ref_sigma_acc = 25
        self.ref_acc_size = 64 # max(w, h)
        self.acc_cov = np.diag(np.array([0.25 * self.dt**4] * 4 + [self.dt**2] * 4, dtype=np.float32))
        self.acc_cov[4:, :4] = np.eye(4, dtype=np.float32) * (0.5 * self.dt**3)
        self.acc_cov[:4, 4:] = np.eye(4, dtype=np.float32) * (0.5 * self.dt**3)
        # self.sigma_acc_size = 2 # smaller acc for w, h
        # self.acc_cov[2:4, 2:4] *= self.sigma_acc_size**2
        # self.acc_cov[2:4, 6:] *= self.sigma_acc_size**2
        # self.acc_cov[6:, 2:4] *= self.sigma_acc_size**2
        # self.acc_cov[6:, 6:] *= self.sigma_acc_size**2

        self.sigma_detection_meas = 1
        self.sigma_flow_meas = 1
        self.init_cov = np.eye(8, dtype=np.float32) * 10
        # self.init_cov[4:6, 4:6] *= 2 # high uncertainty for velocities
        self.init_cov[4:, 4:] *= 2 # high uncertainty for velocities
        self.vel_coupling = 0.6
        self.vel_half_life = 5
        self.max_vel = 5000

        self.prev_frame_gray = None
        self.prev_frame_small = None
        # self.prev_pyramid = None
        self.tracks = OrderedDict()
        self.kalman_filters = {}
        self.flow_tracker = flowtracker.FlowTracker(self.size, estimate_camera_motion=True)

    def track(self, frame, use_flow=True):
        assert self.prev_frame_gray is not None
        assert self.prev_frame_small is not None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.flow_tracker.optflow_scaling[0], fy=self.flow_tracker.optflow_scaling[1])
        self.tracks = OrderedDict(sorted(self.tracks.items(), key=self._compare_dist, reverse=True))
        flow_tracks = deepcopy(self.tracks)
        H_camera = self.flow_tracker.predict(flow_tracks, self.prev_frame_gray, self.prev_frame_small, frame_small)
        if H_camera is not None:
            for track_id, track in list(self.tracks.items()):
                track.frames_since_acquisition += 1
                if track.frames_since_acquisition <= self.n_init:
                    if track_id in flow_tracks:
                        if track.frames_since_acquisition == self.n_init:
                            # initialize kalman filter
                            self.kalman_filters[track_id] = self._create_kalman_filter(track.init_bbox, flow_tracks[track_id].bbox)
                        else:
                            self.tracks[track_id].bbox = flow_tracks[track_id].bbox
                            self.tracks[track_id].feature_pts = flow_tracks[track_id].feature_pts
                            self.tracks[track_id].prev_feature_pts = flow_tracks[track_id].prev_feature_pts
                    else:
                        print('[BBoxTracker] Target lost (flow): %s' % track)
                        del self.tracks[track_id]
                else:
                    # track using kalman filter and flow measurement

                    # only warp during tracking?
                    self._warp_kalman_filters(track_id, H_camera)
                    next_state = self.kalman_filters[track_id].predict()
                    self._clip_vel(track_id)
                    if use_flow and track_id in flow_tracks:
                        self._set_meas_cov(track_id, MeasType.FLOW, flow_tracks[track_id].conf)
                        flow_meas = self._convert_bbox_to_meas(flow_tracks[track_id].bbox)
                        next_state = self.kalman_filters[track_id].correct(flow_meas)
                        self._clip_vel(track_id)
                        self.tracks[track_id].feature_pts = flow_tracks[track_id].feature_pts
                        self.tracks[track_id].prev_feature_pts = flow_tracks[track_id].prev_feature_pts
                    else:
                        self.tracks[track_id].feature_pts = None

                    # check for out of frame case
                    next_bbox = self._convert_state_to_bbox(next_state)
                    inside_bbox = next_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
                    if inside_bbox is not None:
                        self.tracks[track_id].bbox = next_bbox
                        self._set_acc_cov(track_id, next_bbox)
                    else:
                        print('[BBoxTracker] Target lost (outside frame): %s' % track)
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
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.flow_tracker.optflow_scaling[0], fy=self.flow_tracker.optflow_scaling[1])
        for new_track_id, det in enumerate(detections):
            self.tracks[new_track_id] = Track(det.label, det.bbox, new_track_id)
            print('[BBoxTracker] Track registered: %s' % self.tracks[new_track_id])

    def update(self, detections, tile, overlap, acquire=True):
        # filter out tracks and detections not in tile
        sx = sy = 1 - overlap
        scaled_tile = tile.scale(sx, sy)
        tracks = []
        track_ids = []
        boundary_tracks = []
        max_age = self.acquisition_max_age if acquire else self.tracking_max_age
        for track_id, track in self.tracks.items():
            if self.acquire != acquire:
                # reset age when mode toggles
                track.age = 0
            track.age += 1
            if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox): 
                track_ids.append(track_id)
                tracks.append(track)
            elif iou(track.bbox, tile) > 0:
                boundary_tracks.append(track)
        self.acquire = acquire

        # TODO: assign id by label
        # compute optimal assignment using IOU
        all_det_indices = list(range(len(detections)))
        unmatched_det_indices = all_det_indices
        if len(detections) > 0 and len(tracks) > 0:
            iou_mat = np.array([[iou(track.bbox, det.bbox) for det in detections] for track in tracks])
            track_indices, det_indices = linear_sum_assignment(-iou_mat)
            unmatched_det_indices = list(set(all_det_indices) - set(det_indices))
            for track_idx, det_idx in zip(track_indices, det_indices):
                if iou_mat[track_idx, det_idx] > self.min_association_iou:
                    track_id = track_ids[track_idx]
                    if track_id in self.kalman_filters:
                        self._set_meas_cov(track_id, MeasType.CNN)
                        det_meas = self._convert_bbox_to_meas(detections[det_idx].bbox)
                        next_state = self.kalman_filters[track_id].correct(det_meas)
                        self._clip_vel(track_id)
                        next_bbox = self._convert_state_to_bbox(next_state)
                        inside_bbox = next_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
                        if inside_bbox is not None:
                            self.tracks[track_id].bbox = next_bbox
                            self.tracks[track_id].label = detections[det_idx].label
                            self.tracks[track_id].age = 0
                            self._set_acc_cov(track_id, next_bbox)
                        else:
                            print('[BBoxTracker] Target lost (out of frame): %s' % self.tracks[track_id])
                            del self.tracks[track_id]
                            del self.kalman_filters[track_id]
                    else:
                        self.tracks[track_id].bbox = detections[det_idx].bbox
                        self.tracks[track_id].label = detections[det_idx].label
                        self.tracks[track_id].age = 0
                else:
                    unmatched_det_indices.append(det_idx)

        # while min(iou_mat.shape) > 0:
        #     track_idx = iou_mat.max(axis=1).argmax()
        #     det_idx = iou_mat.argmax(axis=1)[track_idx]
        #     if iou_mat[track_idx, det_idx] > self.min_match_iou:
        #         self.tracks[track_ids[track_idx]] = Track(dets[det_idx].label, dets[det_idx].bbox, track_ids[track_idx], dets[det_idx].conf)
        #         iou_mat = np.delete(np.delete(iou_mat, track_idx, axis=0), det_idx, axis=1)
        #         del track_ids[track_idx]
        #         del dets[det_idx]
        #     else:
        #         break

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
                    print('[BBoxTracker] Track registered: %s' % self.tracks[new_track_id])

        # clean up lost tracks
        for track_id, track in list(self.tracks.items()):
            if track.age > max_age:
                print('[BBoxTracker] Target lost (age): %s' % self.tracks[track_id])
                del self.tracks[track_id]
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]

    def get_nearest_track(self):
        if not self.tracks:
            return -1
        nearest_track_id = max(self.tracks.items(), key=self._compare_dist)[0]
        return nearest_track_id

    def _compare_dist(self, id_track_pair):
        # estimate distance by area and y-center
        bin_height = self.size[1] // self.num_vertical_bin
        return (math.ceil(id_track_pair[1].bbox.center()[1] / bin_height), id_track_pair[1].bbox.area())

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

        sigma_acc = max(cur_bbox.size) / self.ref_acc_size * self.ref_sigma_acc
        kalman_filter.processNoiseCov = self.acc_cov * sigma_acc**2
        kalman_filter.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kalman_filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * self.sigma_flow_meas**2
        vels = (np.asarray(cur_bbox.tf_rect()) - np.asarray(init_bbox.tf_rect())) / (self.dt * (self.n_init))
        
        kalman_filter.statePre = np.zeros((8, 1), dtype=np.float32)
        kalman_filter.statePre[:4, 0] = cur_bbox.tf_rect()
        kalman_filter.statePre[4:, 0] = vels
        kalman_filter.statePost = np.empty_like(kalman_filter.statePre, dtype=np.float32)
        np.copyto(kalman_filter.statePost, kalman_filter.statePre)
        kalman_filter.errorCovPost = np.empty_like(self.init_cov, dtype=np.float32)
        np.copyto(kalman_filter.errorCovPost, self.init_cov)

        # # state [x, y, w, h, vx, vy, vw, vh]
        # kalman_filter.transitionMatrix = np.array(
        #     [[1, 0, 0, 0, self.dt, 0, 0, 0],
        #      [0, 1, 0, 0, 0, self.dt, 0, 0], 
        #      [0, 0, 1, 0, 0, 0, self.dt, 0], 
        #      [0, 0, 0, 1, 0, 0, 0, self.dt], 
        #      [0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0, 0], 
        #      [0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0], 
        #      [0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life)]], 
        #     dtype=np.float32
        # )
        
        # sigma_acc = max(cur_bbox.size) / self.ref_acc_size * self.ref_sigma_acc
        # kalman_filter.processNoiseCov = np.empty_like(self.acc_cov, dtype=np.float32)
        # np.copyto(kalman_filter.processNoiseCov, self.acc_cov)
        # kalman_filter.processNoiseCov[:2, :2] *= sigma_acc**2
        # kalman_filter.processNoiseCov[:2, 4:6] *= sigma_acc**2
        # kalman_filter.processNoiseCov[4:6, :2] *= sigma_acc**2
        # kalman_filter.processNoiseCov[4:6, 4:6] *= sigma_acc**2

        # kalman_filter.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        # kalman_filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * self.sigma_flow_meas**2
        # vel = (np.asarray(cur_bbox.center()) - np.asarray(init_bbox.center())) / (self.dt * self.filter_init_frame_count)
        # kalman_filter.statePre = np.zeros((8, 1), dtype=np.float32)
        # kalman_filter.statePre[:4] = self._convert_bbox_to_meas(cur_bbox)
        # kalman_filter.statePre[4:6, 0] = vel
        # kalman_filter.statePost = np.empty_like(kalman_filter.statePre, dtype=np.float32)
        # np.copyto(kalman_filter.statePost, kalman_filter.statePre)
        # kalman_filter.errorCovPost = np.empty_like(self.init_cov, dtype=np.float32)
        # np.copyto(kalman_filter.errorCovPost, self.init_cov)
        return kalman_filter

    def _convert_bbox_to_meas(self, bbox):
        # return np.float32(list(bbox.center()) + list(bbox.size)).reshape(4, 1)
        return np.float32(bbox.tf_rect()).reshape(4, 1)

    def _convert_state_to_bbox(self, state):
        # center = state[:2, 0]
        # size = state[2:4, 0]
        # xmin, ymin = np.int_(np.round(center - (size - 1) / 2))
        # w, h = np.int_(np.round(size))
        # return Rect(cv_rect=(xmin, ymin, w, h))
        return Rect(tf_rect=np.int_(np.round(state[:4, 0])))

    def _set_meas_cov(self, track_id, meas_type=MeasType.FLOW, conf=1.0):
        kalman_filter = self.kalman_filters[track_id]
        if meas_type == MeasType.FLOW:
            kalman_filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * (self.sigma_flow_meas / conf)**2
        elif meas_type == MeasType.CNN:
            kalman_filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * (self.sigma_detection_meas / conf)**2

    def _set_acc_cov(self, track_id, bbox):
        sigma_acc = max(bbox.size) / self.ref_acc_size * self.ref_sigma_acc
        # self.kalman_filters[track_id].processNoiseCov[:2, :2] = self.acc_cov[:2, :2] * sigma_acc**2
        # self.kalman_filters[track_id].processNoiseCov[:2, 4:6] = self.acc_cov[:2, 4:6] * sigma_acc**2
        # self.kalman_filters[track_id].processNoiseCov[4:6, :2] = self.acc_cov[4:6, :2] * sigma_acc**2
        # self.kalman_filters[track_id].processNoiseCov[4:6, 4:6] = self.acc_cov[4:6, 4:6] * sigma_acc**2
        self.kalman_filters[track_id].processNoiseCov = self.acc_cov * sigma_acc**2

    def _clip_vel(self, track_id):
        kalman_filter = self.kalman_filters[track_id]
        kalman_filter.statePost[4:, 0] = np.clip(kalman_filter.statePost[4:, 0], -self.max_vel, self.max_vel) 

    def _warp_kalman_filters(self, track_id, H_camera):
        kalman_filter = self.kalman_filters[track_id]
        pos_tl = kalman_filter.statePost[:2]
        pos_br = kalman_filter.statePost[2:4]
        vel_tl = kalman_filter.statePost[4:6]
        vel_br = kalman_filter.statePost[6:]
        A = H_camera[:2, :2]
        v = H_camera[2, :2].reshape(1, 2)
        t = H_camera[:2, 2].reshape(2, 1)
        # h33 = H_camera[-1, -1]
        # H_grad = lambda p: ((v @ p + 1) * A - (A @ p + t) @ v) / (v @ p + 1)**2
        temp = (v @ pos_tl + 1)
        grad_tl = (temp * A - (A @ pos_tl + t) @ v) / temp**2
        temp = (v @ pos_br + 1)
        grad_br = (temp * A - (A @ pos_br + t) @ v) / temp**2

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
