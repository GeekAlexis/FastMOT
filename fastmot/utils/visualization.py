import colorsys
import numpy as np
import cv2


GOLDEN_RATIO = 0.618033988749895


def draw_tracks(frame, tracks, show_flow=False, show_cov=False):
    for track in tracks:
        draw_bbox(frame, track.tlbr, get_color(track.trk_id), 2, str(track.trk_id))
        if show_flow:
            draw_feature_match(frame, track.prev_keypoints, track.keypoints, (0, 255, 255))
        if show_cov:
            draw_covariance(frame, track.tlbr, track.state[1])


def draw_detections(frame, detections, color=(255, 255, 255), show_conf=False):
    for det in detections:
        text = f'{det.conf:.2f}' if show_conf else None
        draw_bbox(frame, det.tlbr, color, 1, text)


def draw_klt_bboxes(frame, klt_bboxes, color=(0, 0, 0)):
    for tlbr in klt_bboxes:
        draw_bbox(frame, tlbr, color, 1)


def draw_tiles(frame, tiles, scale_factor, color=(0, 0, 0)):
    for tile in tiles:
        tlbr = np.rint(tile * np.tile(scale_factor, 2))
        draw_bbox(frame, tlbr, color, 1)


def draw_background_flow(frame, prev_bg_keypoints, bg_keypoints, color=(0, 0, 255)):
    draw_feature_match(frame, prev_bg_keypoints, bg_keypoints, color)


def get_color(idx, s=0.8, vmin=0.7):
    h = np.fmod(idx * GOLDEN_RATIO, 1.)
    v = 1. - np.fmod(idx * GOLDEN_RATIO, 1. - vmin)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * b), int(255 * g), int(255 * r)


def draw_bbox(frame, tlbr, color, thickness, text=None):
    tlbr = tlbr.astype(int)
    tl, br = tuple(tlbr[:2]), tuple(tlbr[2:])
    cv2.rectangle(frame, tl, br, color, thickness)
    if text is not None:
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(frame, tl, (tl[0] + text_width - 1, tl[1] + text_height - 1),
                      color, cv2.FILLED)
        cv2.putText(frame, text, (tl[0], tl[1] + text_height - 1), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, 0, 1, cv2.LINE_AA)


def draw_feature_match(frame, prev_pts, cur_pts, color):
    if len(cur_pts) > 0:
        cur_pts = np.rint(cur_pts).astype(np.int32)
        for pt in cur_pts:
            cv2.circle(frame, tuple(pt), 1, color, cv2.FILLED)
        if len(prev_pts) > 0:
            prev_pts = np.rint(prev_pts).astype(np.int32)
            for pt1, pt2 in zip(prev_pts, cur_pts):
                cv2.line(frame, tuple(pt1), tuple(pt2), color, 1, cv2.LINE_AA)


def draw_covariance(frame, tlbr, covariance):
    tlbr = tlbr.astype(int)
    tl, br = tuple(tlbr[:2]), tuple(tlbr[2:])

    def ellipse(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        # 95% confidence ellipse
        vals, vecs = np.sqrt(vals[order] * 5.9915), vecs[:, order]
        axes = int(vals[0] + 0.5), int(vals[1] + 0.5)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        return axes, angle

    axes, angle = ellipse(covariance[:2, :2])
    cv2.ellipse(frame, tl, axes, angle, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    axes, angle = ellipse(covariance[2:4, 2:4])
    cv2.ellipse(frame, br, axes, angle, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)


class Visualizer:
    def __init__(self,
                 draw_detections=False,
                 draw_confidence=False,
                 draw_covariance=False,
                 draw_klt=False,
                 draw_obj_flow=False,
                 draw_bg_flow=False):
        """Class for visualization.

        Parameters
        ----------
        draw_detections : bool, optional
            Enable drawing detections.
        draw_confidence : bool, optional
            Enable drawing detection confidence, ignored if `draw_detections` is disabled.
        draw_covariance : bool, optional
            Enable drawing Kalman filter position covariance.
        draw_klt : bool, optional
            Enable drawing KLT bounding boxes.
        draw_obj_flow : bool, optional
            Enable drawing object flow matches.
        draw_bg_flow : bool, optional
            Enable drawing background flow matches.
        """
        self.draw_detections = draw_detections
        self.draw_confidence = draw_confidence
        self.draw_covariance = draw_covariance
        self.draw_klt = draw_klt
        self.draw_obj_flow = draw_obj_flow
        self.draw_bg_flow = draw_bg_flow

    def render(self, frame, tracks, detections, klt_bboxes, prev_bg_keypoints, bg_keypoints):
        """Render visualizations onto the frame."""
        draw_tracks(frame, tracks, show_flow=self.draw_obj_flow, show_cov=self.draw_covariance)
        if self.draw_detections:
            draw_detections(frame, detections, show_conf=self.draw_confidence)
        if self.draw_klt:
            draw_klt_bboxes(frame, klt_bboxes)
        if self.draw_bg_flow:
            draw_background_flow(frame, prev_bg_keypoints, bg_keypoints)
