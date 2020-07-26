from pathlib import Path
import itertools
import json
import math

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

        self.gftt_target_feat_params = Flow.config['gftt_target_feat_params']
        self.fast_bkg_feat_thresh = Flow.config['fast_bkg_feat_thresh']
        self.optflow_params = Flow.config['optflow_params']
        
        self.bkg_feature_detector = cv2.FastFeatureDetector_create(threshold=self.fast_bkg_feat_thresh)

        # background feature points for visualization
        self.bkg_keypoints = np.empty((0, 2), np.float32)
        self.prev_bkg_keypoints = np.empty((0, 2), np.float32)

        # previous frames
        self.prev_frame_gray = None
        self.prev_frame_small = None

        # preallocate masks
        self.ones = np.full(self.size[::-1], 255, np.uint8)
        self.bkg_mask = np.empty_like(self.ones)
        self.fg_mask = self.bkg_mask # alias
        self.frame_rect = Rect(tlwh=(0, 0, *self.size))

    def initiate(self, frame):
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.optflow_scaling[0],
            fy=self.optflow_scaling[1])

    def predict(self, frame, tracks):
        """
        Predict track locations in the current frame using optical flow.
        Parameters
        ----------
        tracks : Dict[int, Track]
            A dictionary with track IDs as keys and tracks as values.
            Feature points of each track are updated in place.
        frame : ndarray
            Current frame.
        Returns
        -------
        Dict[int, Rect]
            Returns a dictionary with track IDs as keys and predicted bounding
            boxes as values.
        """
        # preprocess frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.optflow_scaling[0], fy=self.optflow_scaling[1])
        # order tracks from closest to farthest
        sorted_tracks = sorted(tracks.values(), reverse=True)

        # tic = time.perf_counter()

        # detect target feature points
        # feature_time = 0
        # rect_filter_time = 0
        # feature_pre_time = 0
        # ellipse_filter_time = 0
        all_prev_pts = []
        np.copyto(self.bkg_mask, self.ones)
        for track in sorted_tracks:
            # tic2 = time.perf_counter()
            inside_bbox = track.bbox & self.frame_rect
            keypoints = self._rect_filter(track.keypoints, inside_bbox.tl, inside_bbox.br)
            # rect_filter_time += (time.perf_counter() - tic2)
            if len(keypoints) / inside_bbox.area < self.feature_density:
                # only detect new keypoints when too few are propagated
                # tic2 = time.perf_counter()
                img = inside_bbox.crop(self.prev_frame_gray)
                target_mask = inside_bbox.crop(self.bkg_mask)
                feature_dist = self._gftt_feature_dist(target_mask, self.feature_dist_factor)
                # feature_pre_time += (time.perf_counter() - tic2)
                # tic2 = time.perf_counter()
                keypoints = cv2.goodFeaturesToTrack(img, mask=target_mask, minDistance=feature_dist, 
                    **self.gftt_target_feat_params)
                # feature_time += (time.perf_counter() - tic2)
                if keypoints is None or len(keypoints) == 0:
                    keypoints = np.empty((0, 2), np.float32)
                else:
                    keypoints = self._ellipse_filter(keypoints, track.bbox.center, track.bbox.size, inside_bbox.tl)
                # ellipse_filter_time += (time.perf_counter() - tic2)
            # batch target keypoints
            all_prev_pts.append(keypoints)
            # zero out track in background mask
            inside_bbox.crop(self.bkg_mask)[:] = 0
        target_ends = list(itertools.accumulate(len(pts) for pts in all_prev_pts))
        target_begins = [0] + target_ends[:-1]
        # print('target feature func:', feature_time)
        # print('rect filter:', rect_filter_time)
        # print('feature pre:', feature_pre_time)
        # print('ellipse filter:', ellipse_filter_time)

        # print('target feature:', time.perf_counter() - tic)

        # tic = time.perf_counter()
        # detect background feature points
        if self.estimate_camera_motion:
            prev_frame_small_bkg = cv2.resize(self.prev_frame_gray, None, fx=self.bkg_feature_scaling[0], 
                fy=self.bkg_feature_scaling[1])
            bkg_mask_small = cv2.resize(self.bkg_mask, None, fx=self.bkg_feature_scaling[0],
                fy=self.bkg_feature_scaling[1], interpolation=cv2.INTER_NEAREST)
            keypoints = self.bkg_feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask_small)
            if keypoints is None or len(keypoints) == 0:
                keypoints = np.empty((0, 2), np.float32)
            else:
                keypoints = np.float32([kp.pt for kp in keypoints])
                # term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                # keypoints = cv2.cornerSubPix(prev_frame_small_bkg, keypoints, (5, 5), (-1, -1), term)
                keypoints = self._unscale_pts(keypoints, self.bkg_feature_scaling, None)
            bkg_begin = target_ends[-1]
            all_prev_pts.append(keypoints)
        # print('bg feature:', time.perf_counter() - tic)

        # tic = time.perf_counter()
        # match features using optical flow
        all_prev_pts = np.concatenate(all_prev_pts)
        scaled_prev_pts = self._scale_pts(all_prev_pts, self.optflow_scaling)
        # tic2 = time.perf_counter()
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, frame_small, 
            scaled_prev_pts, None, **self.optflow_params)
        # print('opt flow func:', time.perf_counter() - tic2)
        status = self._get_status(status, err, self.optflow_err_thresh)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.optflow_scaling, status)
        # print('opt flow:', time.perf_counter() - tic)

        # reuse preprocessed frame for next prediction
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small

        # tic = time.perf_counter()
        # estimate camera motion
        H_camera = None
        if self.estimate_camera_motion:
            prev_bkg_pts, matched_bkg_pts = self._get_good_match(all_prev_pts, all_cur_pts, status, bkg_begin, -1)
            if len(matched_bkg_pts) >= 4:
                # H_camera, inlier_mask = cv2.estimateAffinePartial2D(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                #   maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                    maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                self.prev_bkg_keypoints, self.bkg_keypoints = self._get_inliers(prev_bkg_pts, matched_bkg_pts, inlier_mask)
                if H_camera is None or len(self.bkg_keypoints) < self.min_bkg_inlier_count:
                    self.bkg_keypoints = np.empty((0, 2), np.float32)
                    print('[Flow] Background registration failed')
                    return {}, None
                # H_camera = np.vstack((H_camera, [0, 0, 1]))
            else:
                self.bkg_keypoints = np.empty((0, 2), np.float32)
                print('[Flow] Background registration failed')
                return {}, None
        # print('camera homography:', time.perf_counter() - tic)

        # tic = time.perf_counter()
        # affine_time = 0
        # fg_time = 0
        # estimate_time = 0
        # indexing_time = 0
        # post_time = 0
        # estimate target bounding boxes
        # tic2 = time.perf_counter()
        next_bboxes = {}
        np.copyto(self.fg_mask, self.ones)
        # post_time += (time.perf_counter() - tic2)
        for begin, end, track in zip(target_begins, target_ends, sorted_tracks):
            # tic2 = time.perf_counter()
            prev_pts, matched_pts = self._get_good_match(all_prev_pts, all_cur_pts, status, begin, end)
            # indexing_time += (time.perf_counter() - tic2)
            # tic2 = time.perf_counter()
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, self.fg_mask, self.size)
            # fg_time += (time.perf_counter() - tic2)
            if len(matched_pts) < 3:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            # tic2 = time.perf_counter()
            H_affine, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts, method=cv2.RANSAC, 
                maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            # affine_time += (time.perf_counter() - tic2)
            if H_affine is None:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            # tic2 = time.perf_counter()
            est_bbox = Rect(tlwh=self._estimate_bbox(track.bbox.tl, track.bbox.size, H_affine))
            # delete track when it goes outside the frame
            inside_bbox = est_bbox & self.frame_rect
            # estimate_time += (time.perf_counter() - tic2)
            if inside_bbox is None:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            # tic2 = time.perf_counter()
            track.prev_keypoints, track.keypoints = self._get_inliers(prev_pts, matched_pts, inlier_mask)
            next_bboxes[track.trk_id] = est_bbox
            # zero out current track in foreground mask
            est_bbox.crop(self.fg_mask)[:] = 0
            # post_time += (time.perf_counter() - tic2)
        # print('target affine:', time.perf_counter() - tic)
        # print('target affine func:', affine_time)
        # print('indexing:', indexing_time)
        # print('fg filter:', fg_time)
        # print('estimate bbox:', estimate_time)
        # print('post proc:', post_time)
        return next_bboxes, H_camera

    def draw_bkg_feature_match(self, frame):
        if len(self.bkg_keypoints) > 0:
            cur_pts = np.int_(np.rint(self.bkg_keypoints))
            [cv2.circle(frame, tuple(pt), 1, (0, 0, 255), -1) for pt in cur_pts]
            if len(self.prev_bkg_keypoints) > 0:
                prev_pts = np.int_(np.rint(self.prev_bkg_keypoints))
                [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
                    zip(prev_pts, cur_pts)]
            
    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _gftt_feature_dist(target_mask, feature_dist_factor):
        target_area = np.count_nonzero(target_mask)
        est_feat_dist = round(math.sqrt(target_area) * feature_dist_factor)
        return max(est_feat_dist, 1)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_bbox(tl, size, H_affine):
        xmin, ymin = transform(tl, H_affine).ravel()
        scale = math.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        scale = 1. if scale < 0.9 or scale > 1.1 else scale
        return xmin, ymin, scale * size[0], scale * size[1]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tl, br):
        if len(pts) == 0:
            return pts
        ge_le = (pts >= tl) & (pts <= br)
        keep = np.where(ge_le[:, 0] & ge_le[:, 1])
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
        if len(cur_pts) == 0:
            return prev_pts, cur_pts
        size = np.asarray(frame_size)
        quantized_pts = np.rint(cur_pts).astype(np.int_)
        # filter out points outside the frame
        lt = quantized_pts < size
        inside = lt[:, 0] & lt[:, 1]
        prev_pts, cur_pts = prev_pts[inside], cur_pts[inside]
        quantized_pts = quantized_pts[inside]
        # filter out points not in the foreground
        keep = np.array([i for i in range(len(quantized_pts)) if
            fg_mask[quantized_pts[i][1], quantized_pts[i][0]] != 0])
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
            idx = np.where(mask)
            pts[idx] = pts[idx] / scale_factor
        return pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_status(status, err, err_thresh):
        return status.ravel().astype(np.bool_) & (err.ravel() < err_thresh)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_good_match(prev_pts, cur_pts, status, begin, end):
        keep = np.where(status[begin:end])
        prev_pts = prev_pts[begin:end][keep]
        cur_pts = cur_pts[begin:end][keep]
        return prev_pts, cur_pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_inliers(prev_pts, cur_pts, inlier_mask):
        keep = np.where(inlier_mask.ravel())
        return prev_pts[keep], cur_pts[keep]
