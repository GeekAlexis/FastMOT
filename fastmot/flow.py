import logging
import itertools
import numpy as np
import numba as nb
import cv2

from .utils.rect import *


class Flow:
    def __init__(self, size, config):
        self.size = size
        self.bkg_feat_scale_factor = config['bkg_feat_scale_factor']
        self.optflow_scale_factor = config['optflow_scale_factor']
        self.feature_density = config['feature_density']
        self.max_error = config['max_error']
        self.feat_dist_factor = config['feat_dist_factor']
        self.ransac_max_iter = config['ransac_max_iter']
        self.ransac_conf = config['ransac_conf']
        self.min_inlier = config['min_inlier']

        self.bkg_feat_thresh = config['bkg_feat_thresh']
        self.target_feat_params = config['target_feat_params']
        self.optflow_params = config['optflow_params']
        
        self.bkg_feature_detector = cv2.FastFeatureDetector_create(threshold=self.bkg_feat_thresh)

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

        self.frame_rect = to_tlbr((0, 0, *self.size))

    def initiate(self, frame):
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None, fx=self.optflow_scale_factor[0],
            fy=self.optflow_scale_factor[1])

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
        frame_small = cv2.resize(frame_gray, None, fx=self.optflow_scale_factor[0], fy=self.optflow_scale_factor[1])
        # order tracks from closest to farthest
        sorted_tracks = sorted(tracks.values(), reverse=True)

        # detect target feature points
        all_prev_pts = []
        np.copyto(self.bkg_mask, self.ones)
        for track in sorted_tracks:
            inside_tlbr = intersection(track.tlbr, self.frame_rect)
            keypoints = self._rect_filter(track.keypoints, inside_tlbr, self.bkg_mask)
            target_mask = crop(self.bkg_mask, inside_tlbr)
            target_area = mask_area(target_mask)
            if target_area == 0:
                keypoints = np.empty((0, 2), np.float32)
            elif len(keypoints) / target_area < self.feature_density:
                # only detect new keypoints when too few are propagated
                img = crop(self.prev_frame_gray, inside_tlbr)
                # target_mask = crop(self.bkg_mask, inside_tlbr)
                feature_dist = self._estimate_feature_dist(target_area, self.feat_dist_factor)
                keypoints = cv2.goodFeaturesToTrack(img, mask=target_mask, minDistance=feature_dist, 
                    **self.target_feat_params)
                if keypoints is None or len(keypoints) == 0:
                    keypoints = np.empty((0, 2), np.float32)
                else:
                    keypoints = self._ellipse_filter(keypoints, track.tlbr, inside_tlbr[:2])
            # batch target keypoints
            all_prev_pts.append(keypoints)
            # zero out track in background mask
            target_mask[:] = 0
        target_ends = list(itertools.accumulate(len(pts) for pts in all_prev_pts)) if all_prev_pts else [0]
        target_begins = [0] + target_ends[:-1]

        # detect background feature points
        prev_frame_small_bkg = cv2.resize(self.prev_frame_gray, None, fx=self.bkg_feat_scale_factor[0], 
            fy=self.bkg_feat_scale_factor[1])
        bkg_mask_small = cv2.resize(self.bkg_mask, None, fx=self.bkg_feat_scale_factor[0],
            fy=self.bkg_feat_scale_factor[1], interpolation=cv2.INTER_NEAREST)
        keypoints = self.bkg_feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask_small)
        if keypoints is None or len(keypoints) == 0:
            keypoints = np.empty((0, 2), np.float32)
        else:
            keypoints = np.float32([kp.pt for kp in keypoints])
            keypoints = self._unscale_pts(keypoints, self.bkg_feat_scale_factor, None)
        bkg_begin = target_ends[-1]
        all_prev_pts.append(keypoints)

        # match features using optical flow
        all_prev_pts = np.concatenate(all_prev_pts)
        scaled_prev_pts = self._scale_pts(all_prev_pts, self.optflow_scale_factor)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, frame_small, 
            scaled_prev_pts, None, **self.optflow_params)
        status = self._get_status(status, err, self.max_error)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.optflow_scale_factor, status)

        # reuse preprocessed frame for next prediction
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small

        # estimate camera motion
        H_camera = None
        prev_bkg_pts, matched_bkg_pts = self._get_good_match(all_prev_pts, all_cur_pts, status, bkg_begin, -1)
        if len(matched_bkg_pts) >= 4:
            H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, 
                maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            self.prev_bkg_keypoints, self.bkg_keypoints = self._get_inliers(prev_bkg_pts, matched_bkg_pts, inlier_mask)
            if H_camera is None or len(self.bkg_keypoints) < self.min_inlier:
                self.bkg_keypoints = np.empty((0, 2), np.float32)
                logging.warning('Background registration failed')
                return {}, None
        else:
            self.bkg_keypoints = np.empty((0, 2), np.float32)
            logging.warning('Background registration failed')
            return {}, None

        # estimate target bounding boxes
        next_bboxes = {}
        np.copyto(self.fg_mask, self.ones)
        for begin, end, track in zip(target_begins, target_ends, sorted_tracks):
            prev_pts, matched_pts = self._get_good_match(all_prev_pts, all_cur_pts, status, begin, end)
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, self.fg_mask, self.size)
            if len(matched_pts) < 3:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            H_affine, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts, method=cv2.RANSAC, 
                maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            if H_affine is None:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            # delete track when it goes outside the frame
            est_tlbr = self._estimate_bbox(track.tlbr, H_affine)
            track.prev_keypoints, track.keypoints = self._get_inliers(prev_pts, matched_pts, inlier_mask)
            if intersection(est_tlbr, self.frame_rect) is None or len(track.keypoints) < self.min_inlier:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            target_mask = crop(self.fg_mask, est_tlbr)
            next_bboxes[track.trk_id] = est_tlbr
            # zero out current track in foreground mask
            target_mask[:] = 0
        return next_bboxes, H_camera

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_feature_dist(target_area, feat_dist_factor):
        est_feat_dist = round(np.sqrt(target_area) * feat_dist_factor)
        return max(est_feat_dist, 1)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_bbox(tlbr, H_affine):
        tl = transform(tlbr[:2], H_affine).ravel()
        scale = np.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        scale = 1. if scale < 0.9 or scale > 1.1 else scale
        size = scale * get_size(tlbr)
        return to_tlbr(np.append(tl, size))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tlbr, fg_mask):
        if len(pts) == 0 or tlbr is None:
            return np.empty((0, 2), np.float32)
        quantized_pts = np.rint(pts).astype(np.int_)
        mask = np.zeros_like(fg_mask)
        crop(mask, tlbr)[:] = crop(fg_mask, tlbr)
        keep = np.array([i for i in range(len(quantized_pts)) if
            mask[quantized_pts[i][1], quantized_pts[i][0]] != 0])
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _ellipse_filter(pts, tlbr, offset):
        offset = np.asarray(offset, np.float32)
        pts = pts.reshape(-1, 2)
        pts = pts + offset
        center = get_center(tlbr)
        semi_axes = get_size(tlbr) * 0.5
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
        scale_factor = np.asarray(scale_factor, np.float32)
        pts = pts * scale_factor
        pts = pts.reshape(-1, 1, 2)
        return pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True) 
    def _unscale_pts(pts, scale_factor, mask):
        scale_factor = np.asarray(scale_factor, np.float32)
        pts = pts.reshape(-1, 2)
        if mask is None:
            pts = pts / scale_factor
        else:
            idx = np.where(mask)
            pts[idx] = pts[idx] / scale_factor
        return pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_status(status, err, max_err):
        return status.ravel().astype(np.bool_) & (err.ravel() < max_err)

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
