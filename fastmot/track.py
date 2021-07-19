import threading
import numpy as np
import numba as nb

from .models import LABEL_MAP
from .utils.distance import cdist
from .utils.numba import apply_along_axis
from .utils.rect import get_center


class ClusterFeature:
    def __init__(self, num_clusters, metric):
        self.num_clusters = num_clusters
        self.metric = metric
        self.clusters = None
        self.cluster_sizes = None
        self._next_idx = 0

    def __len__(self):
        return self._next_idx

    def update(self, embedding):
        if self._next_idx < self.num_clusters:
            if self.clusters is None:
                self.clusters = np.empty((self.num_clusters, len(embedding)), embedding.dtype)
                self.cluster_sizes = np.zeros(self.num_clusters, np.int_)
            self.clusters[self._next_idx] = embedding
            self.cluster_sizes[self._next_idx] += 1
            self._next_idx += 1
        # elif np.sum(self.cluster_sizes) < 2 * self.num_clusters:
        #     idx = np.random.randint(0, self.num_clusters - 1)
        #     self.cluster_sizes[idx] += 1
        #     self._seq_kmeans(self.clusters, self.cluster_sizes, embedding, idx)
        else:
            nearest_idx = self._get_nearest_cluster(self.clusters, embedding, self.metric)
            self.cluster_sizes[nearest_idx] += 1
            self._seq_kmeans(self.clusters, self.cluster_sizes, embedding, nearest_idx)

    def distance(self, embeddings):
        return self._nearest_cluster_dist(self.clusters, embeddings, self.metric)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_nearest_cluster(clusters, embedding, metric):
        return np.argmin(cdist(np.atleast_2d(embedding), clusters, metric))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _nearest_cluster_dist(clusters, embeddings, metric):
        return apply_along_axis(np.min, cdist(clusters, embeddings, metric), axis=0)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _seq_kmeans(clusters, cluster_sizes, embedding, idx):
        clusters[idx] += (embedding - clusters[idx]) / cluster_sizes[idx]


class SmoothFeature:
    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.smooth_feat = None

    def get(self):
        return self.smooth_feat

    def update(self, embedding):
        if self.smooth_feat is None:
            self.smooth_feat = embedding.copy()
        else:
            self._rolling(self.smooth_feat, embedding, self.lr)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rolling(smooth_feat, embedding, lr):
        smooth_feat = (1. - lr) * smooth_feat + lr * embedding
        smooth_feat /= np.linalg.norm(smooth_feat)


class Track:
    _count = 0
    _lock = threading.Lock()

    def __init__(self, frame_id, tlbr, state, label, metric, num_clusters=4, learning_rate=0.9):
        self.trk_id = self.next_id()
        self.start_frame = frame_id
        self.tlbr = tlbr
        self.state = state
        self.label = label

        self.age = 0
        self.hits = 0
        self.clust_feat = ClusterFeature(num_clusters, metric)
        self.smooth_feat = SmoothFeature(learning_rate)
        self.last_feat = None

        self.inlier_ratio = 1.
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def __str__(self):
        coord = get_center(self.tlbr).astype(int)
        return f'{LABEL_MAP[self.label]} {self.trk_id:>3} at ({coord[0]:>4}, {coord[1]:>3})'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # ordered by approximate distance to the image plane, closer is greater
        return (self.tlbr[-1], -self.age) < (other.tlbr[-1], -other.age)

    @property
    def active(self):
        return self.age < 2

    @property
    def confirmed(self):
        return self.hits > 0

    def update(self, tlbr, state, embedding=None):
        self.tlbr = tlbr
        self.state = state
        if embedding is not None:
            self.age = 0
            self.hits += 1
            self.update_feature(embedding)

    def reinstate(self, frame_id, tlbr, state, embedding):
        self.start_frame = frame_id
        self.tlbr = tlbr
        self.state = state
        self.age = 0
        self.update_feature(embedding)
        self.keypoints = np.empty((0, 2), np.float32)
        self.prev_keypoints = np.empty((0, 2), np.float32)

    def mark_missed(self):
        self.age += 1

    def update_feature(self, embedding):
        self.last_feat = embedding
        self.clust_feat.update(embedding)
        self.smooth_feat.update(embedding)

    @staticmethod
    def reset():
        # with Track._lock:
        Track._count = 0

    @staticmethod
    def next_id():
        # with Track._lock:
        Track._count += 1
        return Track._count
