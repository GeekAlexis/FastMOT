from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cv2

from .utils import InferenceBackend
from .models import *


class FeatureExtractor:
    def __init__(self):
        self.model = OSNet025
        self.batch_size = 32
        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = InferenceBackend(self.model, self.batch_size)
        self.pool = ThreadPool()

    def __call__(self, frame, detections):
        targets = [det.bbox.crop(frame) for det in detections]
        
        cur_targets = []
        embeddings = []
        for offset in range(0, len(targets), self.batch_size):
            cur_targets = targets[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_targets))
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                embeddings.append(embedding_out)
            self.backend.infer_async()
        embedding_out = self.backend.synchronize()[0][:len(cur_targets) * self.feature_dim]
        embeddings.append(embedding_out)

        embeddings = np.reshape(embeddings, (-1, self.feature_dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    @property
    def metric(self):
        return self.model.METRIC

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        img = self._normalize(img)
        self.backend.memcpy(img, idx)

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img):
        # BGR to RGB
        img = img[..., ::-1]
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        img = img * (1 / 255)
        img[0, ...] = (img[0, ...] - 0.485) / 0.229
        img[1, ...] = (img[1, ...] - 0.456) / 0.224
        img[2, ...] = (img[2, ...] - 0.406) / 0.225
        return img.ravel()

# from utils import InferenceBackend
# from models import *
# import os
# import time
# from multiprocessing.pool import ThreadPool

# pool = ThreadPool()


# # inp = np.ones((1, 3, 256, 128))
# # backend = InferenceBackend(OSNet, 1)
# # backend.memcpy_batch(inp.ravel())
# # backend.infer()
# # trt_out = backend.synchronize()

# # onnx_out = np.load('onnx_output.npy')

# # from scipy.spatial.distance import cdist
# # print(cdist(onnx_out, trt_out, 'cosine'))
# # np.testing.assert_allclose(trt_out, onnx_out, rtol=1e-03, atol=1e-05)

# # imgs = [cv2.imread('../test/' + f) for f in os.listdir('../test/')]

# # imgs = []
# # for i, f in enumerate(os.listdir('../test/')):
# #     print(i, f)
# #     imgs.append(cv2.imread('../test/' + f))

# imgs = [np.ones((100, 50, 3)) for _ in range(32)]

# # img1 = cv2.imread('../test/target_1_8.jpg')
# # img2 = cv2.imread('../test/target_1_12.jpg')


# # img1 = cv2.imread('../reid_images/same/target_1_8.jpg')
# # img2 = cv2.imread('../reid_images/same/target_2_9.jpg')

# # img1 = cv2.imread('../reid_images/same/target_6_5.jpg')
# # img2 = cv2.imread('../reid_images/same/target_15_6.jpg')

# # img1 = cv2.imread('../reid_images/same/target_6_8.jpg')
# # img2 = cv2.imread('../reid_images/same/target_15_12.jpg')

# # img1 = cv2.imread('../reid_images/different/target_4_3.jpg')
# # img2 = cv2.imread('../reid_images/different/target_13_11.jpg')
# # img1 = cv2.imread('../reid_images/different/target_14_7.jpg')
# # img2 = cv2.imread('../reid_images/different/target_13_0.jpg')
# # img2 = cv2.imread('../reid_images/different/target_15_1.jpg')

# # # cv2.imwrite('img1_resize.jpg', img1)
# # img1 = preproc(img1)
# # backend = InferenceBackend(OSNet025, 32)
# # backend.memcpy(img1.ravel(), 0)
# # # out1 = backend.infer_v2()[0]

# # img2 = preproc(img2)
# # backend.memcpy(img2.ravel(), 1)
# # backend.infer_async_v2()
# # out = backend.synchronize()[0]

# # out1 = np.asarray(out[:512]).copy()
# # out1 /= np.linalg.norm(out1)
# # out2 = np.asarray(out[512:1024]).copy()
# # out2 /= np.linalg.norm(out2)



# # backend = InferenceBackend(OSNet025, 32)
# # backend.infer_async()
# # out = backend.synchronsize()

# def preproc(img):
#     img = cv2.resize(img, (128, 256))
#     img = _normalize(img)
#     # img = img[..., ::-1]
#     # img = img.transpose(2, 0, 1)

#     # img = img * (1 / 255)
#     # img[0, ...] = (img[0, ...] - 0.485) / 0.229
#     # img[1, ...] = (img[1, ...] - 0.456) / 0.224
#     # img[2, ...] = (img[2, ...] - 0.406) / 0.225
#     # img = img * (2 / 255) - 1
#     # backend.memcpy(img, i)

# @nb.njit(fastmath=True, nogil=True, cache=True)
# def _normalize(img):
#     # BGR to RGB
#     img = img[..., ::-1]
#     # HWC -> CHW
#     img = img.transpose(2, 0, 1)
#     # Normalize using ImageNet's mean and std
#     img = img * (1 / 255)
#     img[0, ...] = (img[0, ...] - 0.485) / 0.229
#     img[1, ...] = (img[1, ...] - 0.456) / 0.224
#     img[2, ...] = (img[2, ...] - 0.406) / 0.225
#     return img.ravel()

# pool.map(preproc, imgs)
# pool.map(preproc, imgs)

# tic = time.perf_counter()
# pool.map(preproc, imgs)
# # for img in imgs:
# #     preproc(img)
# print('preproc', time.perf_counter() - tic)

# # tic = time.perf_counter()
# # backend.infer_async()
# # out = backend.synchronize()[0]
# # print(time.perf_counter() - tic)

# # features = [out[i:i + 512] for i in range(0, len(imgs) * 512, 512)]
# # features /= np.linalg.norm(features, axis=1, keepdims=True)

# # # print(out2)
# # from scipy.spatial.distance import cdist
# # # print(cdist(out1[None, ...], out2[None, ...], 'euclidean'))

# # print(cdist(features, features, 'euclidean'))