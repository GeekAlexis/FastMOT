from collections import deque
import numpy as np

from .models import COCO_LABELS
from .utils.rect import to_tlwh


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
        return "%s %d at %s" % (COCO_LABELS[self.label], self.trk_id, to_tlwh(self.tlbr).astype(int))

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1] // self.bin_height, -self.age) < (other.tlbr[-1] // self.bin_height, -other.age)

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

    def reactivate(self, tlbr):
        raise NotImplementedError
           