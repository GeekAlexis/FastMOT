from collections import deque
import numpy as np
import cv2

from .models import COCO_LABELS


class Track:
    def __init__(self, label, bbox, track_id, feature_buf_size=10):
        self.label = label
        self.bbox = bbox
        self.init_bbox = bbox
        self.track_id = track_id
        self.feature_buf_size = feature_buf_size

        self.age = 0
        self.frames_since_acquired = 0
        self.confirmed = False
        self.alpha = 0.9
        # self.features = deque([], maxlen=self.feature_buf_size)
        self.smooth_feature = None
        self.state = None
        self.feature_pts = None
        self.prev_feature_pts = None

    def __repr__(self):
        return "Track(label=%r, bbox=%r, track_id=%r, feature_buf_size=%r)" % (self.label, self.bbox,
            self.track_id, self.feature_buf_size)

    def __str__(self):
        return "%s ID%d at %s" % (COCO_LABELS[self.label], self.track_id, self.bbox.tlwh)

    def update_features(self, embedding):
        embedding /= np.linalg.norm(embedding)
        if self.smooth_feature is None:
            self.smooth_feature = embedding
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * embedding
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)
        # self.features.append(embedding)

    def draw(self, frame, follow=False, draw_feature_match=False):
        bbox_color = (127, 255, 0) if follow else (0, 165, 255)
        text_color = (143, 48, 0)
        # text = "%s%d" % (COCO_LABELS[self.label], self.track_id) 
        text = str(self.track_id)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), bbox_color, 2)
        cv2.rectangle(frame, tuple(self.bbox.tl), (self.bbox.xmin + text_width - 1,
                        self.bbox.ymin - text_height + 1), bbox_color, cv2.FILLED)
        cv2.putText(frame, text, tuple(self.bbox.tl), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        if draw_feature_match:
            if self.feature_pts is not None:
                [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in np.intc(np.rint(self.feature_pts))]
                if self.prev_feature_pts is not None:
                    [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in 
                        zip(np.intc(np.rint(self.prev_feature_pts)), np.intc(np.rint(self.feature_pts)))]
