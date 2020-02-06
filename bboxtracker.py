import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import time
import math
from copy import deepcopy
from collections import OrderedDict
from util import *


class Track:
    def __init__(self, label, bbox, track_id, det_conf):
        self.label = label
        self.bbox = bbox
        self.track_id = track_id
        self.det_conf = det_conf
        self.age = 0
        self._feature_pts = None
        self._prev_feature_pts = None

    def __repr__(self):
        return "Track(label=%r, bbox=%r, track_id=%r, det_conf=%r)" % (self.label, self.bbox, self.track_id, self.det_conf)

    def __str__(self):
        return "%s ID%d at %s" % (coco_labels[self.label], self.track_id, self.bbox.cv_rect())

    def draw(self, frame, follow=False, draw_feature_match=False):
        bbox_color = (127, 255, 0) if follow else (0, 165, 255)
        text_color = (143, 48, 0)
        text = "%s%d" % (coco_labels[self.label], self.track_id) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), bbox_color, 2)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), bbox_color, cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        if draw_feature_match:
            if self._feature_pts is not None:
                [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in np.int_(np.round(self._feature_pts))]
            if self._prev_feature_pts is not None:
                [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in zip(np.int_(np.round(self._prev_feature_pts)), np.int_(np.round(self._feature_pts)))]


class BBoxTracker:
    def __init__(self, size, estimate_camera_motion=False):
        self.size = size
        self.estimate_camera_motion = estimate_camera_motion
        self.acquisition_max_age = 6
        self.tracking_max_age = 1
        self.bkg_feature_scaling = (0.1, 0.1)
        self.optflow_scaling = (0.5, 0.5)
        self.feature_density = 0.005
        self.min_inlier_count = 5
        self.feature_dist_scaling = 0.06
        self.ransac_maxIter = 20
        self.min_association_iou = 0.3
        self.min_register_conf = 0.6
        self.num_bin = 20
        self.target_feature_detector_type = 'gftt'
        self.bkg_feature_detector_type = 'fast'
        # self.match_thresh = match_thresh

        # parameters for corner detection 
        self.gftt_target_feature_params = dict( 
            maxCorners = 1000,
            qualityLevel = 0.06,
            minDistance = 5,
            blockSize = 3 )
        
        self.gftt_bkg_feature_params = dict( 
            maxCorners = 1000,
            qualityLevel = 0.01,
            minDistance = 5,
            blockSize = 3 )

        self.fast_feature_thresh = 15

        # parameters for lucas kanade optical flow
        self.optflow_params = dict( 
            winSize  = (6, 6),
            maxLevel = 5,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=self.fast_feature_thresh)
        self.prev_frame_gray = None
        self.prev_frame_small = None
        # self.prev_pyramid = None
        self.bkg_feature_pts = None
        self.prev_bkg_feature_pts = None
        self.H_camera = None
        self.tracks = OrderedDict()

    def track(self, frame):
        assert self.prev_frame_gray is not None
        assert self.prev_frame_small is not None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.optflow_scaling[0], fy=self.optflow_scaling[1])
        self.tracks = OrderedDict(sorted(self.tracks.items(), key=self.estimate_dist, reverse=True))

        all_prev_pts = np.empty((0, 2), np.float32)
        target_begin_idices = []
        target_end_idices = []
        bkg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        for track_id, track in list(self.tracks.items()):
            if track._feature_pts is None or len(track._feature_pts) / track.bbox.area() < self.feature_density:
                # TODO: detect more points when target is ocluded
                roi = track.bbox.crop(self.prev_frame_gray)
                if self.target_feature_detector_type == 'gftt':
                    est_minDistance = round(math.sqrt(track.bbox.area()) * self.feature_dist_scaling)
                    self.gftt_target_feature_params['minDistance'] = est_minDistance if est_minDistance > 1 else 1
                    keypoints = cv2.goodFeaturesToTrack(roi, mask=None, **self.gftt_target_feature_params)
                else:
                    keypoints = self.feature_detector.detect(roi, mask=None)
                    keypoints = np.float32([kp.pt for kp in keypoints])
                if keypoints is None or len(keypoints) == 0:
                    if track._feature_pts is not None:
                        keypoints = track._feature_pts
                    else:
                        # keypoints = np.empty((0, 2), dtype=np.float32)
                        del self.tracks[track_id]
                        print('[Tracker] Target lost (no feature points detected): %s' % track)
                        continue
                else:
                    keypoints = keypoints.reshape(-1, 2) + track.bbox.tl()
                    keypoints = self._ellipse_filter(keypoints, track.bbox)
            else:
                keypoints = track._feature_pts
            # scale and batch all target keypoints
            prev_pts = keypoints * self.optflow_scaling
            target_begin_idices.append(len(all_prev_pts))
            all_prev_pts = np.append(all_prev_pts, prev_pts, axis=0)
            target_end_idices.append(len(all_prev_pts))
            # zero out track in background mask
            track.bbox.crop(bkg_mask)[:] = 0

        if self.estimate_camera_motion:
            prev_frame_small_bkg = cv2.resize(self.prev_frame_gray, None, fx=self.bkg_feature_scaling[0], fy=self.bkg_feature_scaling[1])
            bkg_mask = cv2.resize(bkg_mask, None, fx=self.bkg_feature_scaling[0], fy=self.bkg_feature_scaling[1], interpolation=cv2.INTER_NEAREST)
            if self.bkg_feature_detector_type == 'gftt':
                keypoints = cv2.goodFeaturesToTrack(prev_frame_small_bkg, mask=bkg_mask, **self.gftt_bkg_feature_params)
            else:
                keypoints = self.feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask)
                keypoints = np.float32([kp.pt for kp in keypoints])
            if keypoints is not None and len(keypoints) > 0:
                prev_bkg_pts = keypoints.reshape(-1, 2) / self.bkg_feature_scaling * self.optflow_scaling
            else:
                # prev_bkg_pts = np.empty((0, 2), dtype=np.float32)
                self._clear_on_bkg_reg_failure()
                self.prev_frame_gray = frame_gray
                self.prev_frame_small = frame_small
                print('[Tracker] Background registration failed')
                return
            bkg_begin_idx = len(all_prev_pts)
            all_prev_pts = np.append(all_prev_pts, prev_bkg_pts, axis=0)

        # level, pyramid = cv2.buildOpticalFlowPyramid(frame_small, self.optflow_params['winSize'], self.optflow_params['maxLevel'])

        all_prev_pts = np.float32(all_prev_pts).reshape(-1, 1, 2)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, frame_small, all_prev_pts, None, **self.optflow_params)
        # all_prev_pts_r, status, err = cv2.calcOpticalFlowPyrLK(frame_small, self.prev_frame_small, all_cur_pts, None, **self.optflow_params)
        # status_mask = abs(all_prev_pts - all_prev_pts_r).reshape(-1, 2).max(-1) < 1
        status_mask = np.bool_(status) # & (err < self.match_thresh)]
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small
        # self.prev_pyramid = pyramid

        if self.estimate_camera_motion:
            # print(len(all_prev_pts), bkg_begin_idx)
            # print(all_prev_pts[bkg_begin_idx:])
            prev_bkg_pts = all_prev_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            matched_bkg_pts = all_cur_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            if len(matched_bkg_pts) >= 4:
                prev_bkg_pts = prev_bkg_pts / self.optflow_scaling
                matched_bkg_pts = matched_bkg_pts /self.optflow_scaling
                self.H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, maxIters=self.ransac_maxIter)
                if self.H_camera is None or np.count_nonzero(inlier_mask) < self.min_inlier_count:
                    # clear tracks on background reg failure
                    self._clear_on_bkg_reg_failure()
                    print('[Tracker] Background registration failed')
                    return
                else:
                    inlier_mask = np.bool_(inlier_mask.ravel())
                    self.prev_bkg_feature_pts = prev_bkg_pts[inlier_mask].reshape(-1, 2)
                    self.bkg_feature_pts = matched_bkg_pts[inlier_mask].reshape(-1, 2)
            else:
                self._clear_on_bkg_reg_failure()
                print('[Tracker] Background registration failed')
                return

        fg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        for begin, end, (track_id, track) in zip(target_begin_idices, target_end_idices, list(self.tracks.items())):
            prev_pts = all_prev_pts[begin:end][status_mask[begin:end]]
            matched_pts = all_cur_pts[begin:end][status_mask[begin:end]]
            prev_pts = prev_pts / self.optflow_scaling
            matched_pts = matched_pts / self.optflow_scaling
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, fg_mask)
            if len(matched_pts) < 3:
                del self.tracks[track_id]
                print('[Tracker] Target lost (match): %s' % track)
                continue
            H_A, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts, method=cv2.RANSAC, maxIters=self.ransac_maxIter)
            if H_A is None:
                del self.tracks[track_id]
                print('[Tracker] Target lost (inlier): %s' % track)
                continue
            inlier_mask = np.bool_(inlier_mask.ravel())
            track._feature_pts = matched_pts[inlier_mask].reshape(-1, 2)
            track._prev_feature_pts = prev_pts[inlier_mask].reshape(-1, 2)
            est_bbox = self._estimate_bbox(track.bbox, H_A)
            # delete track when it goes outside the frame
            if est_bbox is None:
                del self.tracks[track_id]
                print('[Tracker] Target lost (outside frame): %s' % track)
                continue
            track.bbox = est_bbox
            self.tracks[track_id] = track
            # zero out current track in foreground mask
            track.bbox.crop(fg_mask)[:] = 0

    def init(self, frame, detections):
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.optflow_scaling[0], fy=self.optflow_scaling[1])
        self.tracks = OrderedDict({track_id: Track(det.label, det.bbox, track_id, det.conf) for track_id, det in enumerate(detections)})
        [print('[Tracker] Track registered: %s' % track) for track in self.tracks.values()]

    def update(self, detections, tile, overlap, acquire=True):
        # filter out tracks and detections not in tile
        sx = sy = 1 - overlap
        scaled_tile = tile.scale(sx, sy)
        tracks = []
        track_ids = []
        boundary_tracks = []
        max_age = self.acquisition_max_age if acquire else self.tracking_max_age
        for track_id, track in self.tracks.items():
            self.tracks[track_id].age += 1
            if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox): 
                track_ids.append(track_id)
                tracks.append(track)
            elif iou(track.bbox, tile) > 0:
                boundary_tracks.append(track)
        # dets = deepcopy(detections)

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
                    self.tracks[track_ids[track_idx]] = Track(detections[det_idx].label, detections[det_idx].bbox, track_ids[track_idx], detections[det_idx].conf)
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
                    self.tracks[new_track_id] = Track(detections[det_idx].label, detections[det_idx].bbox, new_track_id, detections[det_idx].conf)
                    print('[Tracker] Track registered: %s' % self.tracks[new_track_id])

        # clean up lost tracks
        for track_id, track in list(self.tracks.items()):
            if track.age > max_age:
                print('[Tracker] Target lost (update): %s' % self.tracks[track_id])
                del self.tracks[track_id]

    def get_nearest_track(self):
        if not self.tracks:
            return -1
        track_id, track = max(self.tracks.items(), key=self.estimate_dist)
        # self.tracks.move_to_end(locked_track_id, last=False)
        return track_id

    def estimate_dist(self, id_track_pair):
        # estimate distance by area and y-center
        bin_height = self.size[1] // self.num_bin
        return (math.ceil(id_track_pair[1].bbox.center()[1] / bin_height), id_track_pair[1].bbox.area())

    def draw_bkg_feature_match(self, frame):
        if self.bkg_feature_pts is not None:
            [cv2.circle(frame, tuple(pt), 1, (0, 0, 255), -1) for pt in np.int_(np.round(self.bkg_feature_pts))]
        if self.prev_bkg_feature_pts is not None:
            [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA) for pt1, pt2 in zip(np.int_(np.round(self.prev_bkg_feature_pts)), np.int_(np.round(self.bkg_feature_pts)))]
    
    def _estimate_bbox(self, bbox, H_A):
        # camera_motion_bbox = None
        # if self.H_camera is not None:
        #     cam_warped = cv2.perspectiveTransform(np.float32([bbox.tl(), bbox.br()]).reshape(2, 1, 2), H_camera)
        #     cam_warped = np.int_(np.round(cam_warped.ravel()))
        #     camera_motion_bbox = Rect(tf_rect=cam_warped)
        warped_tl = cv2.transform(np.float32(bbox.tl()).reshape(1, 1, 2), H_A)
        warped_tl = np.int_(np.round(warped_tl.ravel()))
        s = math.sqrt(H_A[0, 0]**2 + H_A[1, 0]**2)
        s = 1.0 if s < 0.9 or s > 1.1 else s
        new_bbox = Rect(cv_rect=(warped_tl[0], warped_tl[1], int(round(s * bbox.size[0])), int(round(s * bbox.size[1]))))
        new_bbox = new_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
        return new_bbox
        # TODO: kalman filter that takes both camera transform and affine transform

    def _ellipse_filter(self, pts, bbox):
        pts = pts.reshape(-1, 2)
        center = np.asarray(bbox.center())
        axes = np.asarray(bbox.size) * 0.5
        mask = np.sum(((pts - center) / axes)**2, axis=1) <= 1
        return pts[mask]

    def _fg_filter(self, prev_pts, cur_pts, fg_mask):
        # filter out points outside the frame
        cur_pts = np.int_(np.round(cur_pts)).reshape(-1, 2)
        mask = np.all(cur_pts < self.size, axis=1)
        prev_pts, cur_pts = prev_pts[mask], cur_pts[mask]
        # filter out points not on the foreground
        mask = fg_mask[cur_pts[:, 1], cur_pts[:, 0]] == 255
        return prev_pts[mask], cur_pts[mask]

    def _clear_on_bkg_reg_failure(self):
        self.tracks.clear()
        self.bkg_feature_pts = None
        self.prev_bkg_feature_pts = None
        self.H_camera = None
