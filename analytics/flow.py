from pathlib import Path
import json

import numpy as np
import numba as nb
import cv2
import time

from .track import Track
from .utils import transform, Rect, ConfigDecoder


class Flow:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['Flow']

    def __init__(self, size, estimate_camera_motion=False):
        self.size = size
        self.track_targets = estimate_camera_motion
        self.estimate_camera_motion = estimate_camera_motion
        self.bkg_feature_scaling = np.asarray(Flow.config['bkg_feature_scaling'], np.float32)
        self.optflow_scaling = np.asarray(Flow.config['optflow_scaling'], np.float32)
        self.feature_density = Flow.config['feature_density']
        self.optflow_err_thresh = Flow.config['optflow_err_thresh']
        self.min_bkg_inlier_count = Flow.config['min_bkg_inlier_count']
        self.feature_dist_factor = Flow.config['feature_dist_factor']
        self.ransac_max_iter = Flow.config['ransac_max_iter']
        self.ransac_conf = Flow.config['ransac_conf']

        self.gftt_target_feature_params = Flow.config['gftt_target_feature_params']
        self.fast_bkg_feature_thresh = Flow.config['fast_bkg_feature_thresh']
        self.optflow_params = Flow.config['optflow_params']
        # self.gftt_bkg_feature_params = dict( 
        #     maxCorners=1000,
        #     qualityLevel=0.01,
        #     minDistance=5,
        #     blockSize=3
        # )
        # self.optflow_params['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        
        self.fast_feature_detector = cv2.FastFeatureDetector_create(threshold=self.fast_bkg_feature_thresh)
        self.bkg_feature_pts = None
        self.prev_bkg_feature_pts = None

        self.ones = np.full(self.size[::-1], 255, dtype=np.uint8)
        self.bkg_mask = np.empty_like(self.ones)
        self.fg_mask = self.bkg_mask

    def predict(self, tracks, prev_frame_gray, prev_frame_small, frame_small):
        """
        Predict next tracks using optical flow. The function modifies feature points in tracks in place.
        """
        tic = time.perf_counter()
        feature_time = 0

        num_pts = 0
        all_prev_pts = []
        target_begin_idices = []
        target_end_idices = []
        np.copyto(self.bkg_mask, self.ones)
        for track in tracks.values():
            inside_bbox = track.bbox & Rect(0, 0, *self.size)
            if track.feature_pts is not None:
                # only propagate feature points inside the bounding box
                track.feature_pts = self._rect_filter(track.feature_pts, inside_bbox.tl, inside_bbox.br)
            if track.feature_pts is None or len(track.feature_pts) / inside_bbox.area < self.feature_density:
                roi = inside_bbox.crop(prev_frame_gray)
                target_mask = inside_bbox.crop(self.bkg_mask)
                est_min_dist = self._estimate_feature_dist(target_mask, self.feature_dist_factor)
                tic2 = time.perf_counter()
                keypoints = cv2.goodFeaturesToTrack(roi, mask=target_mask, minDistance=est_min_dist, 
                    **self.gftt_target_feature_params)
                feature_time += (time.perf_counter() - tic2)
                if keypoints is None or len(keypoints) == 0:
                    # print('[Flow] Target lost (no corners detected): %s' % track)
                    track.feature_pts = None
                    continue
                else:
                    keypoints = self._ellipse_filter(keypoints, track.bbox.center, track.bbox.size, inside_bbox.tl)
            else:
                keypoints = track.feature_pts
            # scale and batch all target keypoints
            target_begin_idices.append(num_pts)
            all_prev_pts.append(keypoints)
            num_pts += len(keypoints)
            target_end_idices.append(num_pts)
            # zero out track in background mask
            track.bbox.crop(self.bkg_mask)[:] = 0
        # print('target feature:', time.perf_counter() - tic)
        # print('target feature func:', feature_time)

        tic = time.perf_counter()
        if self.estimate_camera_motion:
            prev_frame_small_bkg = cv2.resize(prev_frame_gray, None, fx=self.bkg_feature_scaling[0], 
                fy=self.bkg_feature_scaling[1])
            bkg_mask_small = cv2.resize(self.bkg_mask, None, fx=self.bkg_feature_scaling[0], 
                fy=self.bkg_feature_scaling[1], interpolation=cv2.INTER_NEAREST)
            # keypoints = cv2.goodFeaturesToTrack(prev_frame_small_bkg, mask=bkg_mask, **self.gftt_bkg_feature_params)
            keypoints = self.fast_feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask_small)
            if keypoints is not None and len(keypoints) > 0:
                keypoints = np.float32([kp.pt for kp in keypoints])
                prev_bkg_pts = self._unscale_pts(keypoints, self.bkg_feature_scaling, None)
            else:
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return {}, None
            bkg_begin_idx = num_pts
            all_prev_pts.append(prev_bkg_pts)
        # print('bg feature:', time.perf_counter() - tic)

        # level, pyramid = cv2.buildOpticalFlowPyramid(frame_small, self.optflow_params['winSize'], self.optflow_params['maxLevel'])

        tic = time.perf_counter()
        all_prev_pts = np.concatenate(all_prev_pts)
        all_prev_pts_scaled = self._scale_pts(all_prev_pts, self.optflow_scaling)
        tic2 = time.perf_counter()
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_small, frame_small, 
            all_prev_pts_scaled, None, **self.optflow_params)
        # print('opt flow func:', time.perf_counter() - tic2)
        with np.errstate(invalid='ignore'):
            status_mask = np.bool_(status.ravel()) & (err.ravel() < self.optflow_err_thresh)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.optflow_scaling, status_mask)
        # print('opt flow:', time.perf_counter() - tic)

        tic = time.perf_counter()
        next_bboxes = {}
        H_camera = None
        if self.estimate_camera_motion:
            prev_bkg_pts = all_prev_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            matched_bkg_pts = all_cur_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            if len(matched_bkg_pts) >= 4:
                # H_camera, inlier_mask = cv2.estimateAffinePartial2D(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                #   maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                    maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                inlier_mask = np.bool_(inlier_mask.ravel())
                matched_bkg_inliers = matched_bkg_pts[inlier_mask]
                if H_camera is None or len(matched_bkg_inliers) < self.min_bkg_inlier_count:
                    self.bkg_feature_pts = None
                    self.prev_bkg_feature_pts = None
                    print('[Flow] Background registration failed')
                    return {}, None
                else:
                    # H_camera = np.vstack((H_camera, [0, 0, 1]))
                    self.prev_bkg_feature_pts = prev_bkg_pts[inlier_mask]
                    self.bkg_feature_pts = matched_bkg_inliers
            else:
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return {}, None
        # print('camera homography:', time.perf_counter() - tic)

        tic = time.perf_counter()
        affine_time = 0
        np.copyto(self.fg_mask, self.ones)
        for begin, end, track in zip(target_begin_idices, target_end_idices, tracks.values()):
            prev_pts = all_prev_pts[begin:end][status_mask[begin:end]]
            matched_pts = all_cur_pts[begin:end][status_mask[begin:end]]
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, self.fg_mask, self.size)
            if len(matched_pts) < 3:
                # print('[Flow] Target lost (failed to match): %s' % track)
                track.feature_pts = None
                continue
            tic2 = time.perf_counter()
            H_affine, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts, method=cv2.RANSAC, 
                maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            affine_time += (time.perf_counter() - tic2)
            if H_affine is None:
                # print('[Flow] Target lost (no inlier): %s' % track)
                track.feature_pts = None
                continue
            est_bbox = Rect(*self._estimate_tlwh(track.bbox.tl, track.bbox.size, H_affine))
            # delete track when it goes outside the frame
            inside_bbox = est_bbox & Rect(0, 0, *self.size)
            # inside_bbox = est_bbox.intersect(Rect(0, 0, *self.size))
            if inside_bbox is None:
                # print('[Flow] Target lost (out of frame): %s' % track)
                track.feature_pts = None
                continue
            inlier_mask = np.bool_(inlier_mask.ravel())
            track.feature_pts = matched_pts[inlier_mask]
            track.prev_feature_pts = prev_pts[inlier_mask]
            next_bboxes[track.track_id] = est_bbox
            # zero out current track in foreground mask
            est_bbox.crop(self.fg_mask)[:] = 0
        # print('target affine:', time.perf_counter() - tic)
        # print('target affine func:', affine_time)
        return next_bboxes, H_camera

    def draw_bkg_feature_match(self, frame):
        if self.bkg_feature_pts is not None:
            [cv2.circle(frame, tuple(pt), 1, (0, 0, 255), -1) for pt in np.intc(np.rint(self.bkg_feature_pts))]
        if self.prev_bkg_feature_pts is not None:
            [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
            zip(np.intc(np.rint(self.prev_bkg_feature_pts)), np.intc(np.rint(self.bkg_feature_pts)))]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_feature_dist(target_mask, feature_dist_factor):
        target_area = np.count_nonzero(target_mask)
        est_feat_dist = np.rint(np.sqrt(target_area) * feature_dist_factor)
        return max(est_feat_dist, 1)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_tlwh(tl, size, H_affine):
        xmin, ymin = transform(tl, H_affine).ravel()
        scale = np.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        scale = 1. if scale < 0.9 or scale > 1.1 else scale
        return xmin, ymin, scale * size[0], scale * size[1]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tl, br):
        keep_tl = pts >= tl
        keep_br = pts <= br
        keep = keep_tl[:, 0] & keep_tl[:, 1] & keep_br[:, 0] & keep_br[:, 1]
        # keep = np.all(pts >= bbox.tl, axis=1) & np.all(pts <= bbox.br, axis=1)
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _ellipse_filter(pts, center, axes, offset):
        pts = pts.reshape(-1, 2)
        pts = pts + offset
        semi_axes = np.asarray(axes) * 0.5
        keep = np.sum(((pts - center) / semi_axes)**2, axis=1) <= 1
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fg_filter(prev_pts, cur_pts, fg_mask, frame_size):
        frame_size = np.asarray(frame_size)
        cur_pts_int = np.rint(cur_pts)
        # filter out points outside the frame
        mask = cur_pts_int < frame_size
        mask = mask[:, 0] & mask[:, 1]
        # mask = np.all(cur_pts_int < size, axis=1)
        prev_pts, cur_pts = prev_pts[mask], cur_pts[mask]
        # cur_pts_int = np.intc(cur_pts_int[mask])
        cur_pts_int = cur_pts_int[mask]
        # filter out points not on the foreground
        # keep = fg_mask[cur_pts_int[:, 1], cur_pts_int[:, 0]] != 0
        keep = np.array([i for i in range(len(cur_pts_int)) if 
            fg_mask[int(cur_pts_int[i][1]), int(cur_pts_int[i][0])] != 0])
        return prev_pts[keep], cur_pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _scale_pts(pts, scale_factor):
        pts = pts * scale_factor
        pts = pts.reshape(-1, 1, 2)
        return pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True) 
    def _unscale_pts(pts, scale_factor, mask):
        pts = pts.reshape(-1, 2)
        if mask is None:
            pts = pts / scale_factor
        else:
            pts[mask] = pts[mask] / scale_factor
        return pts
