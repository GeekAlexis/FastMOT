import logging
import itertools
import numpy as np
import numba as nb
import cv2

from .utils.rect import to_tlbr, get_size, get_center
from .utils.rect import mask_area, intersection, crop, transform


LOGGER = logging.getLogger(__name__)


class Flow:
    """
    A KLT tracker based on optical flow feature point matching.
    Camera motion is simultaneously estimated by tracking feature points
    on the background.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    config : Dict
        KLT hyperparameters.
    """

    def __init__(self, size, config):
        self.size = size
        self.bg_feat_scale_factor = config['bg_feat_scale_factor']
        self.opt_flow_scale_factor = config['opt_flow_scale_factor']
        self.feature_density = config['feature_density']
        self.max_error = config['max_error']
        self.feat_dist_factor = config['feat_dist_factor']
        self.ransac_max_iter = config['ransac_max_iter']
        self.ransac_conf = config['ransac_conf']
        self.inlier_thresh = config['inlier_thresh']

        self.bg_feat_thresh = config['bg_feat_thresh']
        self.target_feat_params = config['target_feat_params']
        self.opt_flow_params = config['opt_flow_params']

        self.bg_feat_detector = cv2.FastFeatureDetector_create(threshold=self.bg_feat_thresh)

        # background feature points for visualization
        self.bg_keypoints = np.empty((0, 2), np.float32)
        self.prev_bg_keypoints = np.empty((0, 2), np.float32)

        # previous frames
        self.prev_frame_gray = None
        self.prev_frame_small = None

        # preallocate
        self.ones = np.full(self.size[::-1], 255, np.uint8)
        self.fg_mask = np.empty_like(self.ones)
        self.frame_rect = to_tlbr((0, 0, *self.size))

    def initiate(self, frame):
        """
        Preprocesses the first frame to prepare for subsequent optical
        flow computations.
        Parameters
        ----------
        frame : ndarray
            Initial frame.
        """
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(self.prev_frame_gray, None,
                                           fx=self.opt_flow_scale_factor[0],
                                           fy=self.opt_flow_scale_factor[1])

    def predict(self, frame, tracks):
        """
        Predicts tracklet positions in the next frame and estimates camera motion.
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
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, None, fx=self.opt_flow_scale_factor[0],
                                 fy=self.opt_flow_scale_factor[1])
        # order tracks from closest to farthest
        tracks.sort(reverse=True)

        # detect target feature points
        all_prev_pts = []
        np.copyto(self.fg_mask, self.ones)
        for track in tracks:
            inside_tlbr = intersection(track.tlbr, self.frame_rect)
            target_mask = crop(self.fg_mask, inside_tlbr)
            target_area = mask_area(target_mask)
            keypoints = self._rect_filter(track.keypoints, inside_tlbr, self.fg_mask)
            # only detect new keypoints when too few are propagated
            if len(keypoints) < self.feature_density * target_area:
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
        prev_frame_small_bg = cv2.resize(self.prev_frame_gray, None,
                                         fx=self.bg_feat_scale_factor[0],
                                         fy=self.bg_feat_scale_factor[1])
        bg_mask_small = cv2.resize(self.fg_mask, None, fx=self.bg_feat_scale_factor[0],
                                   fy=self.bg_feat_scale_factor[1], interpolation=cv2.INTER_NEAREST)
        keypoints = self.bg_feat_detector.detect(prev_frame_small_bg, mask=bg_mask_small)
        if len(keypoints) == 0:
            self.bg_keypoints = np.empty((0, 2), np.float32)
            LOGGER.warning('Camera motion estimation failed')
            return {}, None
        keypoints = np.float32([kp.pt for kp in keypoints])
        keypoints = self._unscale_pts(keypoints, self.bg_feat_scale_factor, None)
        bg_begin = target_ends[-1]
        all_prev_pts.append(keypoints)

        # match features using optical flow
        all_prev_pts = np.concatenate(all_prev_pts)
        scaled_prev_pts = self._scale_pts(all_prev_pts, self.opt_flow_scale_factor)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_small, frame_small,
                                                            scaled_prev_pts, None,
                                                            **self.opt_flow_params)
        status = self._get_status(status, err, self.max_error)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.opt_flow_scale_factor, status)

        # reuse preprocessed frame for next prediction
        self.prev_frame_gray = frame_gray
        self.prev_frame_small = frame_small

        # estimate camera motion
        homography = None
        prev_bg_pts, matched_bg_pts = self._get_good_match(all_prev_pts, all_cur_pts,
                                                           status, bg_begin, -1)
        if len(matched_bg_pts) == 0:
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
        np.copyto(self.fg_mask, self.ones)
        for begin, end, track in zip(target_begins, target_ends, tracks):
            prev_pts, matched_pts = self._get_good_match(all_prev_pts, all_cur_pts,
                                                         status, begin, end)
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, self.fg_mask, self.size)
            if len(matched_pts) == 0:
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
        return to_tlbr(np.append(tl, size))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tlbr, fg_mask):
        if len(pts) == 0:
            return np.empty((0, 2), np.float32)
        tl, br = tlbr[:2], tlbr[2:]
        pts2i = np.rint(pts).astype(np.int32)
        # filter out points outside the rectangle
        ge_le = (pts2i >= tl) & (pts2i <= br)
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
        pts = pts.reshape(-1, 2)
        pts = pts + offset
        center = get_center(tlbr)
        semi_axes = get_size(tlbr) * 0.5
        # filter out points outside the ellipse
        keep = np.sum(((pts - center) / semi_axes)**2, axis=1) <= 1.
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fg_filter(prev_pts, cur_pts, fg_mask, frame_size):
        if len(cur_pts) == 0:
            return prev_pts, cur_pts
        size = np.asarray(frame_size)
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
