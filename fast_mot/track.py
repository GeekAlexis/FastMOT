from collections import deque
import numpy as np
import cv2

from .models import COCO_LABELS


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
    def __init__(self, label, bbox, trk_id, feature_buf_size=10):
        self.label = label
        self.bbox = bbox
        self.init_bbox = bbox
        self.trk_id = trk_id
        self.feature_buf_size = feature_buf_size

        self.bin_height = 10
        self.alpha = 0.25 # change this? dynamic?

        self.age = 0
        self.frames_since_acquired = 0
        self.confirmed = False
        self.features = deque([], maxlen=self.feature_buf_size)
        self.smooth_feature = None
        self.state = None

        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __repr__(self):
        return "Track(label=%r, bbox=%r, trk_id=%r, feature_buf_size=%r)" % (self.label, self.bbox,
            self.trk_id, self.feature_buf_size)

    def __str__(self):
        return "%s ID%d at %s" % (COCO_LABELS[self.label], self.trk_id, self.bbox.tlwh)

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.bbox.ymax // self.bin_height, -self.age) < (other.bbox.ymax // self.bin_height, -other.age)
        # return (self.bbox.ymax // self.bin_height, self.bbox.area) < (other.bbox.ymax // self.bin_height, other.bbox.area)

    def update_features(self, embedding):
        if self.smooth_feature is None:
            self.smooth_feature = embedding
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * embedding
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)
        # self.features.append(embedding)
        # if self.trk_id == 1:
        #     print(cdist(self.features, self.features))

    def draw(self, frame, follow=False, draw_feature_match=False):
        bbox_color = (127, 255, 0) if follow else (0, 165, 255)
        text_color = (143, 48, 0)
        # text = "%s%d" % (COCO_LABELS[self.label], self.trk_id) 
        text = str(self.trk_id)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), bbox_color, 2)
        cv2.rectangle(frame, tuple(self.bbox.tl), (self.bbox.xmin + text_width - 1,
                        self.bbox.ymin - text_height + 1), bbox_color, cv2.FILLED)
        cv2.putText(frame, text, tuple(self.bbox.tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1, cv2.LINE_AA)
        # cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), COLORS[self.trk_id % len(COLORS)], 2)

        if draw_feature_match:
            if len(self.keypoints) > 0:
                cur_pts = np.int_(np.rint(self.keypoints))
                [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in cur_pts]
                if len(self.prev_keypoints) > 0:
                    prev_pts = np.int_(np.rint(self.prev_keypoints))
                    [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
                        zip(prev_pts, cur_pts)]
                    
