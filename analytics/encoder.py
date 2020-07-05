import numpy as np
import numba as nb
import cv2

from .utils import *
from .models import *


class ImageEncoder:
    def __init__(self):
        # initialize parameters
        self.model = OSNet
        self.batch_size = 32
        self.backend = InferenceBackend(self.model, self.batch_size)
        self.input_batch = np.empty((self.batch_size, np.prod(self.model.INPUT_SHAPE)))

    def encode(self, frame, detections):
        self.encode_async(frame, detections)
        return self.postprocess()

    def encode_async(self, frame, detections):
        inp = self.preprocess(frame, detections)
        self.backend.infer_async(inp)
    
    def preprocess(self, frame, detections):
        self.num_detections = len(detections)
        assert self.num_detections <= 32
        for i, det in enumerate(detections):
            roi = det.bbox.crop(frame)
            roi = cv2.resize(roi, self.model.INPUT_SHAPE[:0:-1])
            self.input_batch[i] = self._preprocess(roi)
        return self.input_batch

    def postprocess(self):
        embedding_out = self.backend.synchronize()[0]
        embeddings = [embedding_out[i:i + self.model.OUTPUT_LAYOUT] for i in range(self.num_detections)]
        return embeddings

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _preprocess(frame_tile):
        # BGR to RGB
        frame_tile = frame_tile[..., ::-1]
        # HWC -> CHW
        frame_tile = frame_tile.transpose(2, 0, 1)
        # Normalize to [-1.0, 1.0] interval (expected by model)
        frame_tile = frame_tile * (2 / 255) - 1
        return frame_tile.ravel()