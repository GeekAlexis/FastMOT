import math
import numpy as np
import cv2
from util import *


class Flow:
    def __init__(self, size, estimate_camera_motion=False):
        self.size = size
        self.estimate_camera_motion = estimate_camera_motion
        self.bkg_feature_scaling = (0.1, 0.1)
        self.optflow_scaling = (0.5, 0.5)
        self.feature_density = 0.005
        self.opt_flow_err_thresh = 100
        self.min_bkg_inlier_count = 3
        self.feature_dist_scaling = 0.06
        self.ransac_max_iter = 500
        self.ransac_conf = 0.99

        # parameters for corner detection
        self.gftt_target_feature_params = dict( 
            maxCorners=1000,
            qualityLevel=0.06,
            minDistance=5,
            blockSize=3
        )
        self.gftt_bkg_feature_params = dict( 
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=3
        )
        self.fast_feature_thresh = 15

        # parameters for lucas kanade optical flow
        self.optflow_params = dict(
            winSize=(5, 5),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.fast_feature_detector = cv2.FastFeatureDetector_create(threshold=self.fast_feature_thresh)
        self.bkg_feature_pts = None
        self.prev_bkg_feature_pts = None

    def predict(self, tracks, prev_frame_gray, prev_frame_small, frame_small):
        """
        Predict next tracks using optical flow. The function modifies tracks in place.
        """
        all_prev_pts = np.empty((0, 2), np.float32)
        target_begin_idices = []
        target_end_idices = []
        bkg_mask = np.ones(self.size[::-1], dtype=np.uint8) * 255
        for track_id, track in list(tracks.items()):
            inside_bbox = track.bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
            if track.feature_pts is not None:
                # only propagate feature points inside the bounding box
                track.feature_pts = np.array([pt for pt in track.feature_pts if pt in inside_bbox])
            if track.feature_pts is None or len(track.feature_pts) / inside_bbox.area() < self.feature_density:
                roi = inside_bbox.crop(prev_frame_gray)
                target_mask = inside_bbox.crop(bkg_mask)
                target_area = np.count_nonzero(target_mask)
                self.gftt_target_feature_params['minDistance'] = self._estimate_feature_dist(target_area)
                keypoints = cv2.goodFeaturesToTrack(roi, mask=target_mask, **self.gftt_target_feature_params)
                if keypoints is None or len(keypoints) == 0:
                    del tracks[track_id]
                    # print('[Flow] Target lost (no corners detected): %s' % track)
                    continue
                else:
                    keypoints = keypoints.reshape(-1, 2) + inside_bbox.tl()
                    keypoints = self._ellipse_filter(keypoints, track.bbox)
            else:
                keypoints = track.feature_pts
            # scale and batch all target keypoints
            prev_pts = keypoints * self.optflow_scaling
            target_begin_idices.append(len(all_prev_pts))
            all_prev_pts = np.concatenate((all_prev_pts, prev_pts), axis=0)
            target_end_idices.append(len(all_prev_pts))
            # zero out track in background mask
            track.bbox.crop(bkg_mask)[:] = 0

        if self.estimate_camera_motion:
            prev_frame_small_bkg = cv2.resize(prev_frame_gray, None, fx=self.bkg_feature_scaling[0], fy=self.bkg_feature_scaling[1])
            bkg_mask = cv2.resize(bkg_mask, None, fx=self.bkg_feature_scaling[0], fy=self.bkg_feature_scaling[1], interpolation=cv2.INTER_NEAREST)
            # keypoints = cv2.goodFeaturesToTrack(prev_frame_small_bkg, mask=bkg_mask, **self.gftt_bkg_feature_params)
            keypoints = self.fast_feature_detector.detect(prev_frame_small_bkg, mask=bkg_mask)
            if keypoints is not None and len(keypoints) > 0:
                keypoints = np.float32([kp.pt for kp in keypoints])
                prev_bkg_pts = keypoints.reshape(-1, 2) / self.bkg_feature_scaling * self.optflow_scaling
            else:
                tracks.clear()
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return None
            bkg_begin_idx = len(all_prev_pts)
            all_prev_pts = np.concatenate((all_prev_pts, prev_bkg_pts), axis=0)

        # level, pyramid = cv2.buildOpticalFlowPyramid(frame_small, self.optflow_params['winSize'], self.optflow_params['maxLevel'])

        all_prev_pts = np.float32(all_prev_pts).reshape(-1, 1, 2)
        all_cur_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_small, frame_small, all_prev_pts, None, **self.optflow_params)
        # print(np.max(err[status==1]))
        with np.errstate(invalid='ignore'):
            status_mask = (status == 1) & (err < self.opt_flow_err_thresh)
        # status_mask = np.bool_(status)

        H_camera = None
        if self.estimate_camera_motion:
            # print(len(all_prev_pts), bkg_begin_idx)
            # print(all_prev_pts[bkg_begin_idx:])
            prev_bkg_pts = all_prev_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            matched_bkg_pts = all_cur_pts[bkg_begin_idx:][status_mask[bkg_begin_idx:]]
            if len(matched_bkg_pts) >= 4:
                prev_bkg_pts = prev_bkg_pts / self.optflow_scaling
                matched_bkg_pts = matched_bkg_pts /self.optflow_scaling
                # H_camera, inlier_mask = cv2.estimateAffinePartial2D(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                H_camera, inlier_mask = cv2.findHomography(prev_bkg_pts, matched_bkg_pts, method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
                if H_camera is None or np.count_nonzero(inlier_mask) < self.min_bkg_inlier_count:
                    # clear tracks on background reg failure
                    tracks.clear()
                    self.bkg_feature_pts = None
                    self.prev_bkg_feature_pts = None
                    print('[Flow] Background registration failed')
                    return None
                else:
                    # H_camera = np.concatenate((H_camera, [[0, 0, 1]]), axis=0)
                    inlier_mask = np.bool_(inlier_mask.ravel())
                    self.prev_bkg_feature_pts = prev_bkg_pts[inlier_mask].reshape(-1, 2)
                    self.bkg_feature_pts = matched_bkg_pts[inlier_mask].reshape(-1, 2)
            else:
                tracks.clear()
                self.bkg_feature_pts = None
                self.prev_bkg_feature_pts = None
                print('[Flow] Background registration failed')
                return None

        fg_mask = np.ones(self.size[::-1], dtype=np.uint8) * 255
        for begin, end, (track_id, track) in zip(target_begin_idices, target_end_idices, list(tracks.items())):
            prev_pts = all_prev_pts[begin:end][status_mask[begin:end]]
            matched_pts = all_cur_pts[begin:end][status_mask[begin:end]]
            prev_pts = prev_pts / self.optflow_scaling
            matched_pts = matched_pts / self.optflow_scaling
            prev_pts, matched_pts = self._fg_filter(prev_pts, matched_pts, fg_mask)
            if len(matched_pts) < 3:
                del tracks[track_id]
                # print('[Flow] Target lost (failed to match): %s' % track)
                continue
            H_affine, inlier_mask = cv2.estimateAffinePartial2D(prev_pts, matched_pts, method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            if H_affine is None:
                del tracks[track_id]
                # print('[Flow] Target lost (no inlier): %s' % track)
                continue
            est_bbox = self._estimate_bbox(track.bbox, H_affine)
            # delete track when it goes outside the frame
            inside_bbox = est_bbox & Rect(cv_rect=(0, 0, self.size[0], self.size[1]))
            if inside_bbox is None:
                del tracks[track_id]
                # print('[Flow] Target lost (out of frame): %s' % track)
                continue
            track.bbox = est_bbox
            inlier_mask = np.bool_(inlier_mask.ravel())
            track.feature_pts = matched_pts[inlier_mask].reshape(-1, 2)
            track.prev_feature_pts = prev_pts[inlier_mask].reshape(-1, 2)
            # use inlier ratio as confidence
            # inlier_ratio = len(track.feature_pts) / (end - begin) #len(matched_pts)
            # track.conf = inlier_ratio
            # zero out current track in foreground mask
            track.bbox.crop(fg_mask)[:] = 0
        return H_camera

    def draw_bkg_feature_match(self, frame):
        if self.bkg_feature_pts is not None:
            [cv2.circle(frame, tuple(pt), 1, (0, 0, 255), -1) for pt in np.int_(np.round(self.bkg_feature_pts))]
        if self.prev_bkg_feature_pts is not None:
            [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 1, cv2.LINE_AA) for pt1, pt2 in zip(np.int_(np.round(self.prev_bkg_feature_pts)), np.int_(np.round(self.bkg_feature_pts)))]
    
    def _estimate_feature_dist(self, target_area):
        est_ft_dist = round(math.sqrt(target_area) * self.feature_dist_scaling)
        return max(est_ft_dist, 1)

    def _estimate_bbox(self, bbox, H_affine):
        warped_tl = cv2.transform(np.float32(bbox.tl()).reshape(1, 1, 2), H_affine)
        warped_tl = np.int_(np.round(warped_tl.ravel()))
        s = math.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        s = 1.0 if s < 0.9 or s > 1.1 else s
        # s = max(min(s, 1.1), 0.9)
        new_bbox = Rect(cv_rect=(warped_tl[0], warped_tl[1], int(round(s * bbox.size[0])), int(round(s * bbox.size[1]))))

        # warped_center = cv2.transform(np.float32(bbox.center()).reshape(1, 1, 2), H_affine)
        # warped_center = warped_center.ravel()
        # s = math.sqrt(H_affine[0, 0]**2 + H_affine[1, 0]**2)
        # # s = 1.0 if s < 0.9 or s > 1.1 else s
        # s = max(min(s, 1.1), 0.9)
        # new_size = s * np.asarray(bbox.size)
        # xmin, ymin = np.int_(np.round(warped_center - (new_size - 1) / 2))
        # width, height = np.int_(np.round(new_size))
        # new_bbox = Rect(cv_rect=(xmin, ymin, width, height))

        # warped = cv2.transform(np.float32([bbox.tl(), bbox.br()]).reshape(2, 1, 2), H_affine)
        # warped = np.int_(np.round(warped.ravel()))
        # new_bbox = Rect(tf_rect=warped)
        return new_bbox

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
