import numpy as np
import cv2


TRK_COLOR = (0, 165, 255)
DET_COLORS = [
    (75, 25, 230), 
    (48, 130, 245), 
    (25, 225, 255), 
    (60, 245, 210), 
    (75, 180, 60), 
    (240, 240, 70), 
    (200, 130, 0), 
    (180, 30, 145), 
    (230, 50, 240)
]


def draw_trk(frame, trk, draw_feature_match=False):
    tlbr = trk.tlbr.astype(int)
    tl, br = tlbr[:2], tlbr[2:]

    text = str(trk.trk_id)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.rectangle(frame, tuple(tl), tuple(br), TRK_COLOR, 2)
    cv2.rectangle(frame, tuple(tl), (tl[0] + text_width - 1, tl[1] - text_height + 1), TRK_COLOR, cv2.FILLED)
    cv2.putText(frame, text, tuple(tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, (143, 48, 0), 1, cv2.LINE_AA)
    # cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), COLORS[self.trk_id % len(COLORS)], 2)

    if draw_feature_match:
        if len(trk.keypoints) > 0:
            cur_pts = np.rint(trk.keypoints).astype(int)
            [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in cur_pts]
            if len(trk.prev_keypoints) > 0:
                prev_pts = np.rint(trk.prev_keypoints).astype(int)
                [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
                    zip(prev_pts, cur_pts)]


def draw_det(frame, det):
    tlbr = det.tlbr.astype(int)
    tl, br = tlbr[:2], tlbr[2:]
    text = f'{det.conf:.2f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.rectangle(frame, tuple(tl), tuple(br), DET_COLORS[det.tile_id], 2)
    cv2.rectangle(frame, tuple(tl), (tl[0] + text_width - 1, tl[1] - text_height + 1), DET_COLORS[det.tile_id], cv2.FILLED)
    cv2.putText(frame, text, tuple(tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, 0, 1, cv2.LINE_AA)
