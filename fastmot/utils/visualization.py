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


def draw_detections(frame, detections):
    for det in detections:
        draw_bbox(frame, det.tlbr, (255, 255, 255), 1)


def draw_klt_bboxes(frame, tracker):
    for tlbr in tracker.klt_bboxes.values():
        draw_bbox(frame, tlbr, 0, 1)


def draw_tiles(frame, detector):
    assert hasattr(detector, 'tiles')
    for tile in detector.tiles:
        tlbr = np.rint(tile * np.tile(detector.scale_factor, 2))
        draw_bbox(frame, tlbr, 0, 1)


def draw_background_flow(frame, tracker):
    draw_feature_match(frame, tracker.flow.prev_bg_keypoints,
                       tracker.flow.bg_keypoints, (0, 0, 255))


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
