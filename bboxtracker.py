import numpy as np
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
        self._feature_pts = None
        self._prev_feature_pts = None

    def __repr__(self):
        return "Track(label=%r, bbox=%r, track_id=%r, det_conf=%r)" % (self.label, self.bbox, self.track_id, self.det_conf)

    def __str__(self):
        return "%s ID%d at %s" % (coco_labels[self.label], self.track_id, self.bbox.cv_rect())

    def draw(self, frame):
        text = "%s%d" % (coco_labels[self.label], self.track_id) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), (0, 165, 255), 2)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), (0, 165, 255), cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, (143, 48, 0), 2, cv2.LINE_AA)
        if self._feature_pts is not None:
            [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in np.int_(np.round(self._feature_pts))]
        if self._prev_feature_pts is not None:
            [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in zip(np.int_(np.round(self._prev_feature_pts)), np.int_(np.round(self._feature_pts)))]


class BBoxTracker:
    def __init__(self, size, estimate_camera_motion=False):
        self.size = size
        self.estimate_camera_motion = estimate_camera_motion
        self.feature_density = 0.005
        self.bkg_feature_scaling = (0.1, 0.1)
        self.optflow_scaling = (0.5, 0.5)
        self.feature_density = 0.005
        self.min_inlier_count = 5
        self.max_target_area = 80000
        self.feature_dist_scaling = 0.06
        self.ransac_maxIter = 20
        self.fast_feature_thresh = 10
        self.num_bin = 20
        self.min_match_iou = 0.3
        # self.match_thresh = match_thresh

        # parameters for ShiTomasi corner detection
        self.target_feature_params = dict( 
            maxCorners = 1000,
            qualityLevel = 0.1,
            minDistance = 5,
            blockSize = 3 )
        
        self.bkg_feature_params = dict( 
            maxCorners = 1000,
            qualityLevel = 0.01,
            minDistance = 7,
            blockSize = 3 )

        # parameters for lucas kanade optical flow
        self.optflow_params = dict( 
            winSize  = (7, 7),
            maxLevel = 3,
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
        for track in self.tracks.values():
            if track._feature_pts is None or len(track._feature_pts) / track.bbox.area() < self.feature_density:
                # print("detect features")
                roi = track.bbox.crop(self.prev_frame_gray)
                est_minDistance = round(math.sqrt(track.bbox.area()) * self.feature_dist_scaling)
                self.target_feature_params['minDistance'] = est_minDistance if est_minDistance > 1 else 1
                keypoints = cv2.goodFeaturesToTrack(roi, mask=None, **self.target_feature_params)
                # keypoints = self.feature_detector.detect(roi, mask=None)
                # keypoints = np.float32([kp.pt for kp in keypoints])
                if keypoints is None:
                    if track._feature_pts is not None:
                        keypoints = track._feature_pts
                    else:
                        keypoints = np.empty((0, 2), dtype=np.float32)
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
            keypoints = cv2.goodFeaturesToTrack(prev_frame_small_bkg, mask=bkg_mask, **self.bkg_feature_params)
            # keypoints = self.feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask)
            # keypoints = np.float32([kp.pt for kp in keypoints])
            if keypoints is not None:
                prev_bkg_pts = keypoints.reshape(-1, 2) / self.bkg_feature_scaling * self.optflow_scaling
            else:
                prev_bkg_pts = np.empty((0, 2), dtype=np.float32)
            # keypoints = self.feature_detector.detect(self.prev_frame_small, mask=bkg_mask)
            # prev_bkg_pts = np.float32([kp.pt for kp in keypoints])
            bkg_begin_idx = len(all_prev_pts)
            all_prev_pts = np.append(all_prev_pts, prev_bkg_pts, axis=0)

        # level, pyramid = cv2.buildOpticalFlowPyramid(frame_small, self.optflow_params['winSize'], self.optflow_params['maxLevel'])
        # if self.prev_pyramid is None:
        #     self.prev_pyramid = pyramid

        all_prev_pts = np.float32(all_prev_pts).reshape(-1, 1, 2)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, frame_small, all_prev_pts, None, **self.optflow_params)
        # all_prev_pts_r, status, err = cv2.calcOpticalFlowPyrLK(frame_small, self.prev_frame_small, all_cur_pts, None, **self.optflow_params)
        # status_mask = abs(all_prev_pts - all_prev_pts_r).reshape(-1, 2).max(-1) < 1
        status_mask = np.bool_(status) # & (err < self.match_thresh)]

        if self.estimate_camera_motion:
            # print(all_prev_pts[bkg_begin_idx:])
            prev_bkg_pts = all_prev_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            matched_bkg_pts = all_cur_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            if len(matched_bkg_pts) >= 4:
                prev_bkg_pts = prev_bkg_pts / self.optflow_scaling
                matched_bkg_pts = matched_bkg_pts /self.optflow_scaling
                self.H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, maxIters=self.ransac_maxIter)
                if self.H_camera is None or np.count_nonzero(inlier_mask) < self.min_inlier_count:
                    # clear tracks on background reg failure
                    self.tracks.clear()
                    self.bkg_feature_pts = None
                    self.prev_bkg_feature_pts = None
                    self.H_camera = None
                    self.prev_frame_gray = frame_gray
                    self.prev_frame_small = frame_small
                    print('[Tracker] Background registration failed')
                    return {}
                else:
                    inlier_mask = np.bool_(inlier_mask.ravel())
                    self.prev_bkg_feature_pts = prev_bkg_pts[inlier_mask].reshape(-1, 2)
                    self.bkg_feature_pts = matched_bkg_pts[inlier_mask].reshape(-1, 2)
            else:
                self.tracks.clear()
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                self.H_camera = None
                self.prev_frame_gray = frame_gray
                self.prev_frame_small = frame_small
                print('[Tracker] Background registration failed')
                return {}

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
            track.bbox = self._estimate_bbox(track.bbox, H_A)
            # delete track when area too big
            if track.bbox.area() > self.max_target_area:
                del self.tracks[track_id]
                print('[Tracker] Target lost (area): %s' % track)
                continue
            self.tracks[track_id] = track
            # zero out current track in foreground mask
            track.bbox.crop(fg_mask)[:] = 0
                
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small
        # self.prev_pyramid = pyramid
        return self.tracks

    def init(self, frame, detections):
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.optflow_scaling[0], fy=self.optflow_scaling[1])
        self.tracks = OrderedDict({track_id: Track(det.label, det.bbox, track_id, det.conf) for track_id, det in enumerate(detections)})
        [print('[Tracker] Track registered: %s' % track) for track in self.tracks.values()]

    def update(self, detections, tile, overlap, acquire=True):
        # filter out tracks and detections not in tile
        sx = sy = 1 - overlap
        scaled_tile = tile.scale(sx, sy)
        track_ids = []
        tracks = []
        boundary_tracks = []
        for track_id, track in list(self.tracks.items()):
            if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox): 
                track_ids.append(track_id)
                tracks.append(track)
            elif acquire:
                if iou(track.bbox, tile) > 0:
                    boundary_tracks.append(track)
            else:
                del self.tracks[track_id]
        dets = deepcopy(detections)
        # dets = [det for det in detections if det.bbox.center() in scaled_tile]

        # TODO: assign id by label and use Hungarian algo?
        # compute optimal assignment using IOU
        iou_mat = np.array([[iou(track.bbox, det.bbox) for det in dets] for track in tracks])
        while min(iou_mat.shape) > 0:
            track_idx = iou_mat.max(axis=1).argmax()
            det_idx = iou_mat.argmax(axis=1)[track_idx]
            if iou_mat[track_idx, det_idx] > self.min_match_iou:
                self.tracks[track_ids[track_idx]] = Track(dets[det_idx].label, dets[det_idx].bbox, track_ids[track_idx], dets[det_idx].conf)
                iou_mat = np.delete(np.delete(iou_mat, track_idx, axis=0), det_idx, axis=1)
                del track_ids[track_idx]
                del dets[det_idx]
            else:
                break

        # register new detections
        if acquire:
            for det in dets:
                register = True
                for track in boundary_tracks:
                    if det.label == track.label and iou(det.bbox, track.bbox) > 0.1:
                        register = False
                if register:
                    new_track_id = 0
                    while new_track_id in self.tracks:
                        new_track_id += 1
                    self.tracks[new_track_id] = Track(det.label, det.bbox, new_track_id, det.conf)
                    print('[Tracker] Track registered: %s' % self.tracks[new_track_id])

        # clean up lost tracks
        for track_id in track_ids:
            print('[Tracker] Target lost (update): %s' % self.tracks[track_id])
            del self.tracks[track_id]
        return self.tracks

    def lock_on(self):
        if not self.tracks:
            return -1
        # focus on the closest track only
        locked_track_id, locked_track = max(self.tracks.items(), key=self.estimate_dist)
        self.tracks.move_to_end(locked_track_id, last=False)
        print('[Tracker] Track locked on: %s' % locked_track)
        return locked_track_id

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
