import math
import numpy as np
import cv2


class Detection:
    def __init__(self, bbox, label, conf):
        self.bbox = bbox
        self.label = label
        self.conf = conf

    def __repr__(self):
        return "Detection(bbox=%r, label=%r, conf=%r)" % (self.bbox, self.label, self.conf)

    def __str__(self):
        return "%.2f %s at %s" % (self.conf, coco_labels[self.label], self.bbox.cv_rect())
    
    def draw(self, frame):
        text = "%s: %.2f" % (coco_labels[self.label], self.conf) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), (112, 25, 25), 2)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), (112, 25, 25), cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 255), 2, cv2.LINE_AA)


class Track:
    def __init__(self, label, bbox, track_id):
        self.label = label
        self.bbox = bbox
        self.init_bbox = bbox
        self.track_id = track_id
        self.age = 0
        self.conf = 1
        self.feature_pts = None
        self.prev_feature_pts = None
        self.frames_since_acquired = 0

    def __repr__(self):
        return "Track(label=%r, bbox=%r, track_id=%r)" % (self.label, self.bbox, self.track_id)

    def __str__(self):
        return "%s ID%d at %s" % (coco_labels[self.label], self.track_id, self.bbox.cv_rect())

    def draw(self, frame, follow=False, draw_feature_match=False):
        bbox_color = (127, 255, 0) if follow else (0, 165, 255)
        text_color = (143, 48, 0)
        text = "%s%d" % (coco_labels[self.label], self.track_id) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), bbox_color, 2)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), bbox_color, cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        if draw_feature_match:
            if self.feature_pts is not None:
                [cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1) for pt in np.int_(np.round(self.feature_pts))]
                if self.prev_feature_pts is not None:
                    [cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 255), 1, cv2.LINE_AA) for pt1, pt2 in zip(np.int_(np.round(self.prev_feature_pts)), np.int_(np.round(self.feature_pts)))]


class Rect:
    def __init__(self, tf_rect=None, cv_rect=None):
        if tf_rect is not None:
            self.xmin, self.ymin, self.xmax, self.ymax = tf_rect
            self.size = (self.xmax - self.xmin + 1, self.ymax - self.ymin + 1)
        elif cv_rect is not None:
            self.xmin, self.ymin = cv_rect[:2]
            self.size = tuple(cv_rect[2:])
            self.xmax = self.size[0] - 1 + self.xmin
            self.ymax = self.size[1] - 1 + self.ymin
        
    def __repr__(self):
        return "Rect(tf_rect=(%r, %r, %r, %r))" % (self.xmin, self.ymin, self.xmax, self.ymax)

    def __contains__(self, point):
        assert isinstance(point, (tuple, list, np.ndarray)) and len(point) == 2
        return point[0] >= self.xmin and point[1] >= self.ymin and point[0] <= self.xmax and point[1] <= self.ymax

    def __and__(self, other):
        # intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        inter_rect = Rect(tf_rect=(xmin, ymin, xmax, ymax))
        if inter_rect.size[0] <= 0 or inter_rect.size[1] <= 0:
            return None
        return inter_rect

    def __or__(self, other):
        # minimum rect that contains both rects
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        return Rect(tf_rect=(xmin, ymin, xmax, ymax))

    def contains_rect(self, other):
        return other.xmin >= self.xmin and other.ymin >= self.ymin and other.xmax <= self.xmax and other.ymax <= self.ymax

    def tf_rect(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def cv_rect(self):
        return (self.xmin, self.ymin, self.size[0], self.size[1])

    def tl(self):
        return (self.xmin, self.ymin)

    def br(self):
        return (self.xmax, self.ymax)

    def center(self):
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def corners(self):
        return (self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax)

    def area(self):
        return self.size[0] * self.size[1]

    def crop(self, image):
        return image[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def scale(self, sx, sy):
        half_size = (self.size * np.array([sx, sy]) - 1) / 2
        xmin, ymin = np.int_(np.round(self.center() - half_size))
        xmax, ymax = np.int_(np.round(self.center() + half_size))
        return Rect(tf_rect=(xmin, ymin, xmax, ymax))


def iou(rect1, rect2):
    inter_xmin = max(rect1.xmin, rect2.xmin) 
    inter_ymin = max(rect1.ymin, rect2.ymin)
    inter_xmax = min(rect1.xmax, rect2.xmax)
    inter_ymax = min(rect1.ymax, rect2.ymax)
    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)
    iou = inter_area / (rect1.area() + rect2.area() - inter_area)
    return iou


coco_labels = [
    'unlabeled',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]
