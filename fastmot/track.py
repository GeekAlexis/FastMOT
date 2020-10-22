import numpy as np

from .models import LABEL_MAP
from .utils.rect import get_center


class Track:
    def __init__(self, frame_id, trk_id, tlbr, label):
        self.start_frame = frame_id
        self.trk_id = trk_id
        self.tlbr = tlbr
        self.label = label

        self.alpha = 0.9
        self.age = 0
        self.confirmed = False
        self.smooth_feature = None
        self.state = None

        self.inlier_ratio = 1.
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __str__(self):
        return "%s %d at %s" % (LABEL_MAP[self.label], self.trk_id,
                                get_center(self.tlbr).astype(int))

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1], -self.age) < (other.tlbr[-1], -other.age)

    @property
    def active(self):
        return self.age < 2

    def update_feature(self, embedding):
        if self.smooth_feature is None:
            self.smooth_feature = embedding
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * embedding
            self.smooth_feature /= np.linalg.norm(self.smooth_feature)

    def reactivate(self, frame_id, tlbr, embedding):
        self.start_frame = frame_id
        self.tlbr = tlbr
        self.age = 0
        self.update_feature(embedding)
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)
           