from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cupy as cp
import cupyx.scipy.ndimage
import cv2

from . import models
from .utils import TRTInference
from .utils.rect import multi_crop


class FeatureExtractor:
    def __init__(self, config):
        self.model = getattr(models, config['model'])
        self.batch_size = config['batch_size']

        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)
        self.pool = ThreadPool()

        self.embeddings = []
        self.num_features = 0

        # c, h, w = self.model.INPUT_SHAPE
        # self.small_devs = cp.empty((self.batch_size, h, w, 3), dtype=np.uint8)
        # self.map_streams = [cp.cuda.stream.Stream() for _ in range(self.batch_size)]
        # self.end_events = [cp.cuda.Event() for _ in range(self.batch_size)]

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def __call__(self, frame, detections):
        self.extract_async(frame, detections)
        return self.postprocess()

    @property
    def metric(self):
        return self.model.METRIC

    def extract_async(self, frame, detections):
        """
        Extract feature embeddings from detections asynchronously.
        """
        # frame_dev = cp.asarray(frame)
        imgs = multi_crop(frame, detections.tlbr)
        self.embeddings, cur_imgs = [], []
        # pipeline inference and preprocessing the next batch in parallel
        for offset in range(0, len(imgs), self.batch_size):
            cur_imgs = imgs[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_imgs))
            # self._map(cur_imgs)
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                self.embeddings.append(embedding_out)
            # self.backend.infer_async2()
            self.backend.infer_async()
        self.num_features = len(cur_imgs)

    def postprocess(self):
        """
        Synchronizes, applies postprocessing, and returns a NxM matrix of N
        extracted embeddings with dimension M.
        This function should be called after `extract_async`.
        """
        if self.num_features == 0:
            return np.empty((0, self.feature_dim))

        embedding_out = self.backend.synchronize()[0][:self.num_features * self.feature_dim]
        self.embeddings.append(embedding_out)
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        self._normalize(img, self.inp_handle[idx])

        # img_dev = cp.asarray(img)
        # _, h, w = self.model.INPUT_SHAPE
        # zoom_factors = (h / img_dev.shape[0], w / img_dev.shape[1], 1.)
        # cupyx.scipy.ndimage.zoom(img_dev, zoom_factors, output=self.small_devs[idx], order=1, mode='opencv', grid_mode=True)
        # img_dev = self.small_devs[idx][..., ::-1]
        # img_dev = img_dev.transpose(2, 0, 1)
        # offset = idx * self.inp_stride
        # out = self.backend.input.device[offset:offset + self.inp_stride].reshape(self.model.INPUT_SHAPE)
        # # img_dev *= 1 / 255
        # out[0, ...] = (img_dev[0, ...] / 255. - 0.485) / 0.229
        # out[1, ...] = (img_dev[1, ...] / 255. - 0.456) / 0.224
        # out[2, ...] = (img_dev[2, ...] / 255. - 0.406) / 0.225

    def _map(self, cur_imgs):
        for idx, img in enumerate(cur_imgs):
            with self.map_streams[idx]:
                self._preprocess(idx, img)
            # self.end_events[idx].record(stream)

        for stream in self.map_streams:
            stream.synchronize()

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
