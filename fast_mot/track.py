from collections import deque
import numpy as np
import cv2

from .models import COCO_LABELS
from .utils import *


COLORS = [
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


class Track:
    def __init__(self, tlbr, label, trk_id):
        self.tlbr = tlbr
        self.init_tlbr = tlbr
        self.label = label
        self.trk_id = trk_id

        self.feature_buf_size = 10
        self.bin_height = 10
        self.alpha = 0.7

        self.age = 0
        self.frames_since_acquired = 0
        self.confirmed = False
        self.features = deque([], maxlen=self.feature_buf_size)
        self.smooth_feature = None
        self.state = None

        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __repr__(self):
        return "Track(tlbr=%r, label=%r, trk_id=%r)" % (self.tlbr, self.label, self.trk_id)

    def __str__(self):
        return "%s ID %d at %s" % (COCO_LABELS[self.label], self.trk_id, to_tlwh(self.tlbr).astype(int))

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1] // self.bin_height, -self.age) < (other.tlbr[-1] // self.bin_height, -other.age)
        # return (self.bbox.ymax // self.bin_height, self.bbox.area) < (other.bbox.ymax // self.bin_height, other.bbox.area)

    @property
    def active(self):
        return self.age < 3

    def update_features(self, embedding):
        if self.smooth_feature is None:
            self.smooth_feature = embedding
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * embedding
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)
        # self.features.append(embedding)
        # if self.trk_id == 1:
        #     print(cdist(self.features, self.features))

    def draw(self, frame, draw_feature_match=False):
        tlbr = self.tlbr.astype(int)
        tl, br = tlbr[:2], tlbr[2:]

        bbox_color = (0, 165, 255)
        text_color = (143, 48, 0)
        # text = "%s%d" % (COCO_LABELS[self.label], self.trk_id) 
        text = str(self.trk_id)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(frame, tuple(tl), tuple(br), bbox_color, 2)
        cv2.rectangle(frame, tuple(tl), (tl[0] + text_width - 1, tl[1] - text_height + 1), bbox_color, cv2.FILLED)
        cv2.putText(frame, text, tuple(tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1, cv2.LINE_AA)
        # cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), COLORS[self.trk_id % len(COLORS)], 2)

        if draw_feature_match:
            if len(self.keypoints) > 0:
                cur_pts = np.rint(self.keypoints).astype(int)
                [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in cur_pts]
                if len(self.prev_keypoints) > 0:
                    prev_pts = np.rint(self.prev_keypoints).astype(int)
                    [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
                        zip(prev_pts, cur_pts)]
                    
