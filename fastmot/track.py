import numpy as np

from .models import LABEL_MAP
from .utils.rect import get_center


class Track:
    def __init__(self, tlbr, label, trk_id, frame_id):
        self.tlbr = tlbr
        self.init_tlbr = tlbr
        self.label = label
        self.trk_id = trk_id
        self.start_frame = frame_id

        self.bin_height = 10
        self.alpha = 0.9

        self.age = 0
        self.confirmed = False
        self.smooth_feature = None
        self.state = None
        
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __repr__(self):
        return "Track(tlbr=%r, label=%r, trk_id=%r, frame_id=%r)" % (self.tlbr, self.label, self.trk_id, self.start_frame)

    def __str__(self):
        return "%s %d at %s" % (LABEL_MAP[self.label], self.trk_id, get_center(self.tlbr).astype(int))

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

    def reactivate(self, frame_id, tlbr, embedding):
        self.tlbr = tlbr
        self.init_tlbr = tlbr
        self.start_frame = frame_id
        self.age = 0
        self.update_features(embedding)
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)
           