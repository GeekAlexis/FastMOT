import logging
import itertools
import numpy as np
import numba as nb
import cupyx
import cv2

from .utils.rect import to_tlbr, get_size, get_center
from .utils.rect import mask_area, intersection, crop
from .utils.numba import transform


LOGGER = logging.getLogger(__name__)


class Flow:
    def __init__(self, size,
                 bg_feat_scale_factor=(0.1, 0.1),
                 opt_flow_scale_factor=(0.5, 0.5),
                 feat_density=0.005,
                 feat_dist_factor=0.06,
                 ransac_max_iter=500,
                 ransac_conf=0.99,
                 max_error=100,
                 inlier_thresh=4,
                 bg_feat_thresh=10,
                 target_feat_params=None,
                 opt_flow_params=None):
        """A KLT tracker based on optical flow feature point matching.
        Camera motion is simultaneously estimated by tracking feature points
        on the background.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        bg_feat_scale_factor : tuple, optional
            Width and height scale factors to resize frame for background feature detection.
        opt_flow_scale_factor : tuple, optional
            Width and height scale factors to resize frame for optical flow.
        feat_density : float, optional
            Min feature point density to keep inside the bounding box.
        feat_dist_factor : float, optional
            Target size scale factor to estimate min feature point distance.
        ransac_max_iter : int, optional
            Max RANSAC iterations to filter matched outliers.
        ransac_conf : float, optional
            RANSAC confidence threshold to filter matched outliers.
        max_error : int, optional
            Max optical flow error.
        inlier_thresh : int, optional
            Min number of inliers for valid matching.
        bg_feat_thresh : int, optional
            FAST threshold for background feature detection.
        target_feat_params : SimpleNamespace, optional
            GFTT parameters for target feature detection, see `cv2.goodFeaturesToTrack`.
        opt_flow_params : SimpleNamespace, optional
            Optical flow parameters, see `cv2.calcOpticalFlowPyrLK`.
        """
        self.size = size
        assert 0 < bg_feat_scale_factor[0] <= 1 and 0 < bg_feat_scale_factor[1] <= 1
        self.bg_feat_scale_factor = bg_feat_scale_factor
        assert 0 < opt_flow_scale_factor[0] <= 1 and 0 < opt_flow_scale_factor[1] <= 1
        self.opt_flow_scale_factor = opt_flow_scale_factor
        assert 0 <= feat_density <= 1
        self.feat_density = feat_density
        assert feat_dist_factor >= 0
        self.feat_dist_factor = feat_dist_factor
        assert ransac_max_iter >= 0
        self.ransac_max_iter = ransac_max_iter
        assert 0 <= ransac_conf <= 1
        self.ransac_conf = ransac_conf
        assert 0 <= max_error <= 255
        self.max_error = max_error
        assert inlier_thresh >= 1
        self.inlier_thresh = inlier_thresh
        assert bg_feat_thresh >= 0
        self.bg_feat_thresh = bg_feat_thresh

        self.target_feat_params = {
            "maxCorners": 1000,
            "qualityLevel": 0.06,
            "blockSize": 3
        }
        self.opt_flow_params = {
            "winSize": (5, 5),
            "maxLevel": 5,
            "criteria": (3, 10, 0.03)
        }
        if target_feat_params is not None:
            self.target_feat_params.update(vars(target_feat_params))
        if opt_flow_params is None:
            self.opt_flow_params.update(vars(opt_flow_params))

        self.bg_feat_detector = cv2.FastFeatureDetector_create(threshold=self.bg_feat_thresh)

        # background feature points for visualization
        self.bg_keypoints = None
        self.prev_bg_keypoints = None

        # preallocate frame buffers
        opt_flow_sz = (
            round(self.opt_flow_scale_factor[0] * self.size[0]),
            round(self.opt_flow_scale_factor[1] * self.size[1])
        )
        self.frame_gray = cupyx.empty_pinned(self.size[::-1], np.uint8)
        self.frame_small = cupyx.empty_pinned(opt_flow_sz[::-1], np.uint8)
        self.prev_frame_gray = cupyx.empty_like_pinned(self.frame_gray)
        self.prev_frame_small = cupyx.empty_like_pinned(self.frame_small)

        bg_feat_sz = (
            round(self.bg_feat_scale_factor[0] * self.size[0]),
            round(self.bg_feat_scale_factor[1] * self.size[1])
        )
        self.prev_frame_bg = cupyx.empty_pinned(bg_feat_sz[::-1], np.uint8)
        self.bg_mask_small = cupyx.empty_like_pinned(self.prev_frame_bg)

        self.fg_mask = cupyx.empty_like_pinned(self.frame_gray)
        self.frame_rect = to_tlbr((0, 0, *self.size))

    def init(self, frame):
        """Preprocesses the first frame to prepare for subsequent `predict`.

        Parameters
        ----------
        frame : ndarray
            Initial frame.
        """
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.prev_frame_gray)
        cv2.resize(self.prev_frame_gray, self.prev_frame_small.shape[::-1],
                   dst=self.prev_frame_small)
        self.bg_keypoints = np.empty((0, 2), np.float32)
        self.prev_bg_keypoints = np.empty((0, 2), np.float32)

    def predict(self, frame, tracks):
        """Predicts tracklet positions in the next frame and estimates camera motion.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        tracks : List[Track]
            List of tracks to predict.
            Feature points of each track are updated in place.

        Returns
        -------
        Dict[int, ndarray], ndarray
            Returns a dictionary with track IDs as keys and predicted bounding
            boxes of [x1, x2, y1, y2] as values, and a 3x3 homography matrix.
        """
        # preprocess frame
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.frame_gray)
        cv2.resize(self.frame_gray, self.frame_small.shape[::-1], dst=self.frame_small)

        # order tracks from closest to farthest
        tracks.sort(reverse=True)

        # detect target feature points
        all_prev_pts = []
        self.fg_mask[:] = 255
        for track in tracks:
            inside_tlbr = intersection(track.tlbr, self.frame_rect)
            target_mask = crop(self.fg_mask, inside_tlbr)
            target_area = mask_area(target_mask)
            keypoints = self._rect_filter(track.keypoints, inside_tlbr, self.fg_mask)
            # only detect new keypoints when too few are propagated
            if len(keypoints) < self.feat_density * target_area:
                img = crop(self.prev_frame_gray, inside_tlbr)
                feature_dist = self._estimate_feature_dist(target_area, self.feat_dist_factor)
                keypoints = cv2.goodFeaturesToTrack(img, mask=target_mask,
                                                    minDistance=feature_dist,
                                                    **self.target_feat_params)
                if keypoints is None:
                    keypoints = np.empty((0, 2), np.float32)
                else:
                    keypoints = self._ellipse_filter(keypoints, track.tlbr, inside_tlbr[:2])
            # batch keypoints
            all_prev_pts.append(keypoints)
            # zero out target in foreground mask
            target_mask[:] = 0
        target_ends = list(itertools.accumulate(len(pts) for pts in
                                                all_prev_pts)) if all_prev_pts else [0]
        target_begins = itertools.chain([0], target_ends[:-1])

        # detect background feature points
        cv2.resize(self.prev_frame_gray, self.prev_frame_bg.shape[::-1], dst=self.prev_frame_bg)
        cv2.resize(self.fg_mask, self.bg_mask_small.shape[::-1], dst=self.bg_mask_small,
                   interpolation=cv2.INTER_NEAREST)
        keypoints = self.bg_feat_detector.detect(self.prev_frame_bg, mask=self.bg_mask_small)
        if len(keypoints) == 0:
            self.bg_keypoints = np.empty((0, 2), np.float32)
            self.prev_frame_gray, self.frame_gray = self.frame_gray, self.prev_frame_gray
            self.prev_frame_small, self.frame_small = self.frame_small, self.prev_frame_small
            LOGGER.warning('Camera motion estimation failed')
            return {}, None
        keypoints = np.float32([kp.pt for kp in keypoints])
        keypoints = self._unscale_pts(keypoints, self.bg_feat_scale_factor)
        bg_begin = target_ends[-1]
        all_prev_pts.append(keypoints)

        # match features using optical flow
        all_prev_pts = np.concatenate(all_prev_pts)
        scaled_prev_pts = self._scale_pts(all_prev_pts, self.opt_flow_scale_factor)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, self.frame_small,
                                                            scaled_prev_pts, None,
                                                            **self.opt_flow_params)
        status = self._get_status(status, err, self.max_error)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.opt_flow_scale_factor, status)

        # save preprocessed frame buffers for next prediction
        self.prev_frame_gray, self.frame_gray = self.frame_gray, self.prev_frame_gray
        self.prev_frame_small, self.frame_small = self.frame_small, self.prev_frame_small

        # estimate camera motion
        homography = None
        prev_bg_pts, matched_bg_pts = self._get_good_match(all_prev_pts, all_cur_pts,
                                                           status, bg_begin, -1)
        if len(matched_bg_pts) < 4:
            self.bg_keypoints = np.empty((0, 2), np.float32)
            LOGGER.warning('Camera motion estimation failed')
            return {}, None
        homography, inlier_mask = cv2.findHomography(prev_bg_pts, matched_bg_pts,
                                                     method=cv2.RANSAC,
                                                     maxIters=self.ransac_max_iter,
                                                     confidence=self.ransac_conf)
        self.prev_bg_keypoints, self.bg_keypoints = self._get_inliers(prev_bg_pts, matched_bg_pts,
                                                                      inlier_mask)
        if homography is None or len(self.bg_keypoints) < self.inlier_thresh:
            self.bg_keypoints = np.empty((0, 2), np.float32)
            LOGGER.warning('Camera motion estimation failed')
            return {}, None

        # estimate target bounding boxes
        next_bboxes = {}
        self.fg_mask[:] = 255
        for begin, end, track in zip(target_begins, target_ends, tracks):
            prev_pts, matched_pts = self._get_good_match(all_prev_pts, all_cur_pts,
                                                         status, begin, end)
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, self.fg_mask, self.size)
            if len(matched_pts) < 3:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            # model motion as partial affine
            affine_mat, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts,
                                                                  method=cv2.RANSAC,
                                                                  maxIters=self.ransac_max_iter,
                                                                  confidence=self.ransac_conf)
            if affine_mat is None:
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            est_tlbr = self._estimate_bbox(track.tlbr, affine_mat)
            track.prev_keypoints, track.keypoints = self._get_inliers(prev_pts, matched_pts,
                                                                      inlier_mask)
            if (intersection(est_tlbr, self.frame_rect) is None or
                    len(track.keypoints) < self.inlier_thresh):
                track.keypoints = np.empty((0, 2), np.float32)
                continue
            next_bboxes[track.trk_id] = est_tlbr
            track.inlier_ratio = len(track.keypoints) / len(matched_pts)
            # zero out predicted target in foreground mask
            target_mask = crop(self.fg_mask, est_tlbr)
            target_mask[:] = 0
        return next_bboxes, homography

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_feature_dist(target_area, feat_dist_factor):
        est_feat_dist = round(np.sqrt(target_area) * feat_dist_factor)
        return max(est_feat_dist, 1)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_bbox(tlbr, affine_mat):
        tl = transform(tlbr[:2], affine_mat).ravel()
        scale = np.linalg.norm(affine_mat[:2, 0])
        scale = 1. if scale < 0.9 or scale > 1.1 else scale
        size = scale * get_size(tlbr)
        return to_tlbr((tl[0], tl[1], size[0], size[1]))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tlbr, fg_mask):
        if len(pts) == 0:
            return np.empty((0, 2), np.float32)
        pts2i = np.rint(pts).astype(np.int32)
        # filter out points outside the rectangle
        ge_le = (pts2i >= tlbr[:2]) & (pts2i <= tlbr[2:])
        inside = np.where(ge_le[:, 0] & ge_le[:, 1])
        pts, pts2i = pts[inside], pts2i[inside]
        # keep points inside the foreground area
        keep = np.array([i for i in range(len(pts2i)) if
                         fg_mask[pts2i[i][1], pts2i[i][0]] == 255])
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _ellipse_filter(pts, tlbr, offset):
        offset = np.asarray(offset, np.float32)
        center = np.asarray(get_center(tlbr))
        semi_axes = get_size(tlbr) * 0.5
        pts = pts.reshape(-1, 2)
        pts = pts + offset
        # filter out points outside the ellipse
        keep = np.sum(((pts - center) / semi_axes)**2, axis=1) <= 1.
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fg_filter(prev_pts, cur_pts, fg_mask, frame_sz):
        if len(cur_pts) == 0:
            return prev_pts, cur_pts
        size = np.asarray(frame_sz)
        pts2i = np.rint(cur_pts).astype(np.int32)
        # filter out points outside the frame
        ge_lt = (pts2i >= 0) & (pts2i < size)
        inside = ge_lt[:, 0] & ge_lt[:, 1]
        prev_pts, cur_pts = prev_pts[inside], cur_pts[inside]
        pts2i = pts2i[inside]
        # keep points inside the foreground area
        keep = np.array([i for i in range(len(pts2i)) if
                         fg_mask[pts2i[i][1], pts2i[i][0]] == 255])
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
    def _unscale_pts(pts, scale_factor, mask=None):
        scale_factor = np.asarray(scale_factor, np.float32)
        unscale_factor = 1 / scale_factor
        pts = pts.reshape(-1, 2)
        if mask is None:
            pts = pts * unscale_factor
        else:
            idx = np.where(mask)
            pts[idx] = pts[idx] * unscale_factor
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
