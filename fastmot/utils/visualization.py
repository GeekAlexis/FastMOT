import numpy as np
import cv2


DARK_COLORS = [
    (128, 0, 0),
    (170, 110, 40),
    (128, 128, 0),
    (0, 128, 128),
    (0, 0, 128),
    (0, 0, 0),
]


COLORS = [
    (75, 25, 230), 
    (48, 130, 245), 
    (25, 225, 255), 
    (60, 245, 210), 
    (75, 180, 60), 
    (240, 240, 70), 
    (200, 130, 0), 
    (180, 30, 145), 
    (230, 50, 240),
]


LIGHT_COLORS = [
    (250, 190, 212),
    (255, 215, 180),
    (255, 250, 200),
    (170, 255, 195),
    (220, 190, 255),
    (255, 255, 255),
]


def draw_trk(frame, trk, draw_flow=False):
    _draw_bbox(frame, trk.tlbr, str(trk.trk_id), COLORS[trk.trk_id % len(COLORS)], 0)
    if draw_flow:
        _draw_feature_match(frame, trk.keypoints, trk.prev_keypoints, (0, 255, 255))


def draw_det(frame, det):
    _draw_bbox(frame, det.tlbr, f'{det.conf:.2f}', LIGHT_COLORS[det.label % len(LIGHT_COLORS)], 0)


def draw_tile(frame, detector):
    assert hasattr(detector, 'tiles')
    for tile in detector.tiles:
        tl = np.rint(tile[:2] * detector.scale_factor).astype(int)
        br = np.rint(tile[2:] * detector.scale_factor).astype(int)
        cv2.rectangle(frame, tuple(tl), tuple(br), 0, 1)


def draw_bkg_flow(frame, tracker):
    _draw_feature_match(frame, tracker.flow.bkg_keypoints, tracker.flow.prev_bkg_keypoints, (0, 0, 255))


def _draw_bbox(frame, tlbr, text, bbox_color, text_color):
    tlbr = tlbr.astype(int)
    tl, br = tlbr[:2], tlbr[2:]
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.rectangle(frame, tuple(tl), tuple(br), bbox_color, 2)
    cv2.rectangle(frame, tuple(tl), (tl[0] + text_width - 1, tl[1] - text_height + 1), bbox_color, cv2.FILLED)
    cv2.putText(frame, text, tuple(tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1, cv2.LINE_AA)


def _draw_feature_match(frame, cur_pts, prev_pts, color):
    if len(cur_pts) > 0:
        cur_pts = np.rint(cur_pts).astype(int)
        [cv2.circle(frame, tuple(pt), 1, color, -1) for pt in cur_pts]
        if len(prev_pts) > 0:
            prev_pts = np.rint(prev_pts).astype(int)
            [cv2.line(frame, tuple(pt1), tuple(pt2), color, 1, cv2.LINE_AA) for pt1, pt2 in 
                zip(prev_pts, cur_pts)]