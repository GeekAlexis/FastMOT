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
        self.bkg_feature_scaling = Flow.config['bkg_feature_scaling']
        self.optflow_scaling = Flow.config['optflow_scaling']
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
        all_prev_pts = np.empty((0, 2), np.float32)
        target_begin_idices = []
        target_end_idices = []
        np.copyto(self.bkg_mask, self.ones)
        for track in tracks.values():
            # inside_bbox = track.bbox & Rect(0, 0, *self.size)
            inside_bbox = track.bbox.intersect(Rect(0, 0, *self.size))
            if track.feature_pts is not None:
                # only propagate feature points inside the bounding box
                track.feature_pts = self._rect_filter(track.feature_pts, inside_bbox)
            if track.feature_pts is None or len(track.feature_pts) / inside_bbox.area < self.feature_density:
                roi = inside_bbox.crop(prev_frame_gray)
                target_mask = inside_bbox.crop(self.bkg_mask)
                est_min_dist = self._estimate_feature_dist(target_mask)
                tic2 = time.perf_counter()
                keypoints = cv2.goodFeaturesToTrack(roi, mask=target_mask, minDistance=est_min_dist, 
                                                    **self.gftt_target_feature_params)
                feature_time += (time.perf_counter() - tic2)
                if keypoints is None or len(keypoints) == 0:
                    # print('[Flow] Target lost (no corners detected): %s' % track)
                    track.feature_pts = None
                    continue
                else:
                    keypoints = keypoints + inside_bbox.tl
                    keypoints = self._ellipse_filter(keypoints, track.bbox)
            else:
                keypoints = track.feature_pts
            # scale and batch all target keypoints
            target_begin_idices.append(len(all_prev_pts))
            all_prev_pts = np.vstack((all_prev_pts, keypoints))
            target_end_idices.append(len(all_prev_pts))
            # zero out track in background mask
            track.bbox.crop(self.bkg_mask)[:] = 0
        print('target feature:', time.perf_counter() - tic)
        print('target feature func:', feature_time)

        tic = time.perf_counter()
        if self.estimate_camera_motion:
            prev_frame_small_bkg = cv2.resize(prev_frame_gray, None, fx=self.bkg_feature_scaling[0], 
                                                fy=self.bkg_feature_scaling[1])
            bkg_mask_small = cv2.resize(self.bkg_mask, None, fx=self.bkg_feature_scaling[0], fy=self.bkg_feature_scaling[1],
                                        interpolation=cv2.INTER_NEAREST)
            # keypoints = cv2.goodFeaturesToTrack(prev_frame_small_bkg, mask=bkg_mask, **self.gftt_bkg_feature_params)
            keypoints = self.fast_feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask_small)
            if keypoints is not None and len(keypoints) > 0:
                keypoints = np.float32([kp.pt for kp in keypoints])
                prev_bkg_pts = keypoints / self.bkg_feature_scaling
            else:
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return {}, None
            bkg_begin_idx = len(all_prev_pts)
            all_prev_pts = np.vstack((all_prev_pts, prev_bkg_pts))
        print('bg feature:', time.perf_counter() - tic)

        # level, pyramid = cv2.buildOpticalFlowPyramid(frame_small, self.optflow_params['winSize'], self.optflow_params['maxLevel'])

        # tic = time.perf_counter()
        all_prev_pts = all_prev_pts.reshape(-1, 1, 2)
        all_prev_pts_scaled = np.float32(all_prev_pts * self.optflow_scaling)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_small, frame_small, all_prev_pts_scaled, None, 
                                                            **self.optflow_params)
        # print(np.max(err[status==1]))
        with np.errstate(invalid='ignore'):
            status_mask = np.bool_(status) & (err < self.optflow_err_thresh)
        # print('opt flow:', time.perf_counter() - tic)
        all_cur_pts[status_mask] = all_cur_pts[status_mask] / self.optflow_scaling

        tic = time.perf_counter()
        next_bboxes = {}
        H_camera = None
        if self.estimate_camera_motion:
            prev_bkg_pts = all_prev_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            matched_bkg_pts = all_cur_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            if len(matched_bkg_pts) >= 4:
                # H_camera, inlier_mask = cv2.estimateAffinePartial2D(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                                                            maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                if H_camera is None or np.count_nonzero(inlier_mask) < self.min_bkg_inlier_count:
                    self.bkg_feature_pts = None
                    self.prev_bkg_feature_pts = None
                    print('[Flow] Background registration failed')
                    return {}, None
                else:
                    # H_camera = np.vstack((H_camera, [0, 0, 1]))
                    inlier_mask = np.bool_(inlier_mask.ravel())
                    self.prev_bkg_feature_pts = prev_bkg_pts[inlier_mask].reshape(-1, 2)
                    self.bkg_feature_pts = matched_bkg_pts[inlier_mask].reshape(-1, 2)
            else:
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return {}, None
        print('camera homography:', time.perf_counter() - tic)

        tic = time.perf_counter()
        affine_time = 0
        np.copyto(self.fg_mask, self.ones)
        for begin, end, (track_id, track) in zip(target_begin_idices, target_end_idices, tracks.items()):
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
                continue
            est_bbox = self._estimate_bbox(track.bbox, H_affine)
            # delete track when it goes outside the frame
            # inside_bbox = est_bbox & Rect(0, 0, *self.size)
            inside_bbox = est_bbox.intersect(Rect(0, 0, *self.size))
            if inside_bbox is None:
                # print('[Flow] Target lost (out of frame): %s' % track)
                track.feature_pts = None
                continue
            inlier_mask = np.bool_(inlier_mask.ravel())
            track.feature_pts = matched_pts[inlier_mask].reshape(-1, 2)
            track.prev_feature_pts = prev_pts[inlier_mask].reshape(-1, 2)
            next_bboxes[track_id] = est_bbox
            # zero out current track in foreground mask
            est_bbox.crop(self.fg_mask)[:] = 0
        print('target affine:', time.perf_counter() - tic)
        print('target affine func:', affine_time)
        return next_bboxes, H_camera

    def draw_bkg_feature_match(self, frame):
        if self.bkg_feature_pts is not None:
            [cv2.circle(frame, tuple(pt), 1, (0, 0, 255), -1) for pt in np.intc(np.round(self.bkg_feature_pts))]
        if self.prev_bkg_feature_pts is not None:
            [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
            zip(np.intc(np.round(self.prev_bkg_feature_pts)), np.intc(np.round(self.bkg_feature_pts)))]
    
    def _estimate_feature_dist(self, target_mask):
        target_area = np.count_nonzero(target_mask)
        est_feat_dist = round(np.sqrt(target_area) * self.feature_dist_factor)
        return max(est_feat_dist, 1)

    @staticmethod
    # @nb.njit(fastmath=True)
    def _estimate_bbox(bbox, H_affine):
        # warped_tl = cv2.transform(np.float32(bbox.tl()).reshape(1, 1, 2), H_affine)
        warped_xmin, warped_ymin = transform(bbox.tl, H_affine).ravel()
        s = np.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        s = 1.0 if s < 0.9 or s > 1.1 else s
        new_w, new_h = s * np.asarray(bbox.size)
        new_bbox = Rect(warped_xmin, warped_ymin, new_w, new_h)

        # warped_center = cv2.transform(np.float32(bbox.center()).reshape(1, 1, 2), H_affine)
        # warped_center = warped_center.ravel()
        # s = np.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        # # s = 1.0 if s < 0.9 or s > 1.1 else s
        # s = max(min(s, 1.1), 0.9)
        # new_size = s * np.asarray(bbox.size)
        # xmin, ymin = np.intc(np.round(warped_center - (new_size - 1) / 2))
        # width, height = np.intc(np.round(new_size))
        # new_bbox = Rect(xmin, ymin, width, height)

        # warped = cv2.transform(np.float32([bbox.tl(), bbox.br()]).reshape(2, 1, 2), H_affine)
        # warped = np.intc(np.round(warped.ravel()))
        # new_bbox = Rect(tlbr=warped)
        return new_bbox

    @staticmethod
    # @nb.njit(parallel=True)
    def _rect_filter(pts, bbox):
        return np.array([pt for pt in pts if bbox.contains(pt)])

    @staticmethod
    # @nb.njit(fastmath=True)
    def _ellipse_filter(pts, bbox):
        pts = pts.reshape(-1, 2)
        axes = np.asarray(bbox.size) * 0.5
        mask = np.sum(((pts - bbox.center) / axes)**2, axis=1) <= 1
        return pts[mask]

    @staticmethod
    # @nb.njit(fastmath=True, cache=True)
    def _fg_filter(prev_pts, cur_pts, fg_mask, size):
        size = np.asarray(size)
        cur_pts = cur_pts.reshape(-1, 2)
        out = np.empty_like(cur_pts)
        np.round(cur_pts, 0, out)
        cur_pts = np.intc(out)
        # filter out points outside the frame
        # mask = cur_pts < size
        # mask = mask[:, 0] & mask[:, 1]
        mask = np.all(cur_pts < size, axis=1)
        prev_pts, cur_pts = prev_pts[mask], cur_pts[mask]
        # filter out points not on the foreground
        mask = fg_mask[cur_pts[:, 1], cur_pts[:, 0]] != 0
        # idx = []
        # for i in nb.prange(len(cur_pts)):
        #     pt = quantized_cur_pts[i]
        #     if fg_mask[pt[1], pt[0]] == 255:
        #         idx.append(i)
        return prev_pts[mask], cur_pts[mask]
