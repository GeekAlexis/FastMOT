from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cv2

from . import models
from .utils import TRTInference
from .utils.rect import multi_crop


class FeatureExtractor:
    def __init__(self, model='OSNet025', batch_size=16):
        self.model = getattr(models, model)
        assert batch_size >= 1
        self.batch_size = batch_size

        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)
        self.pool = ThreadPool()

        self.embeddings = []
        self.last_num_features = 0

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def __call__(self, frame, tlbrs):
        self.extract_async(frame, tlbrs)
        return self.postprocess()

    @property
    def metric(self):
        return self.model.METRIC

    def extract_async(self, frame, tlbrs):
        """
        Extract feature embeddings from bounding boxes asynchronously.
        """
        imgs = multi_crop(frame, tlbrs)
        self.embeddings, cur_imgs = [], []
        # pipeline inference and preprocessing the next batch in parallel
        for offset in range(0, len(imgs), self.batch_size):
            cur_imgs = imgs[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_imgs))
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                self.embeddings.append(embedding_out)
            self.backend.infer_async()
        self.last_num_features = len(cur_imgs)

    def postprocess(self):
        """
        Synchronizes, applies postprocessing, and returns a NxM matrix of N
        extracted embeddings with dimension M.
        This API should be called after `extract_async`.
        """
        if self.last_num_features == 0:
            return np.empty((0, self.feature_dim))

        embedding_out = self.backend.synchronize()[0][:self.last_num_features * self.feature_dim]
        self.embeddings.append(embedding_out)
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def null_embeddings(self, detections):
        """
        Returns a NxM matrix of N identical embeddings with dimension M.
        This API effectively disables feature extraction.
        """
        embeddings = np.ones((len(detections), self.feature_dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        self._normalize(img, self.inp_handle[idx])

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img, out):
        # BGR to RGB
        rgb = img[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        out[0, ...] = (chw[0, ...] / 255. - 0.485) / 0.229
        out[1, ...] = (chw[1, ...] / 255. - 0.456) / 0.224
        out[2, ...] = (chw[2, ...] / 255. - 0.406) / 0.225
