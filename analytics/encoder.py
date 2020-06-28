import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

from .utils import *
from .models import *


class ObjectEncoder:
    def __init__(self):
        # initialize parameters
        self.model = OSNet
        self.batch_size = 32
        self.backend = InferenceBackend(self.model, self.batch_size)
        self.input_batch = np.zeros((self.batch_size, np.prod(self.model.INPUT_SHAPE)))
        # self.pool = ThreadPool(processes=4)

    def encode(self, frame, detections):
        self.encode_async(frame, detections)
        return self.postprocess()

    def encode_async(self, frame, detections):
        inp = self.preprocess(frame, detections)
        self.backend.infer_async(inp)
    
    def preprocess(self, frame, detections):
        self.num_detections = len(detections)
        assert self.num_detections <= 32
        # self.frame = frame
        for i, det in enumerate(detections):
            roi = det.bbox.crop(frame)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, self.model.INPUT_SHAPE[:0:-1])
            roi = np.transpose(roi, (2, 0, 1)) # HWC -> CHW
            self.input_batch[i] = roi.ravel()
        # self.pool.starmap(self._preprocess_worker, enumerate(detections))
        self.input_batch[:self.num_detections] = self.input_batch[:self.num_detections] * (2 / 255) - 1
        return self.input_batch

    # def _preprocess_worker(self, idx, det):
    #     roi = det.bbox.crop(self.frame)
    #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    #     roi = cv2.resize(roi, self.model.INPUT_SHAPE[:0:-1])
    #     roi = np.transpose(roi, (2, 0, 1)) # HWC -> CHW
    #     self.input_batch[idx] = roi.ravel()

    def postprocess(self):
        embedding_out = self.backend.synchronize()[0]
        embeddings = [embedding_out[i:i + self.model.OUTPUT_LAYOUT] for i in range(self.num_detections)]
        return embeddings