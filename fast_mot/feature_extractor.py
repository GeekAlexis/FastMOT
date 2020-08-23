from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cv2

from .utils import InferenceBackend, multi_crop
from .models import *


class FeatureExtractor:
    def __init__(self):
        self.model = OSNet025
        self.batch_size = 32
        self.input_size = np.prod(self.model.INPUT_SHAPE)
        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = InferenceBackend(self.model, self.batch_size)
        self.pool = ThreadPool()

        self.embeddings = []
        self.num_features = 0

    def __call__(self, frame, detections):
        # if len(detections) == 0:
        #     return np.empty((0, self.feature_dim))

        # imgs = multi_crop(frame, detections.tlbr)
        # embeddings = []
        # for offset in range(0, len(imgs), self.batch_size):
        #     cur_imgs = imgs[offset:offset + self.batch_size]
        #     self.pool.starmap(self._preprocess, enumerate(cur_imgs))
        #     if offset > 0:
        #         embedding_out = self.backend.synchronize()[0]
        #         embeddings.append(embedding_out)
        #     self.backend.infer_async()

        # embedding_out = self.backend.synchronize()[0][:len(cur_imgs) * self.feature_dim]
        # embeddings.append(embedding_out)

        # embeddings = np.concatenate(embeddings).reshape(-1, self.feature_dim)
        # embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        # return embeddings

        self.extract_async(frame, detections)
        return self.postprocess()

    @property
    def metric(self):
        return self.model.METRIC

    def extract_async(self, frame, detections):
        imgs = multi_crop(frame, detections.tlbr)
        self.embeddings, cur_imgs = [], []
        for offset in range(0, len(imgs), self.batch_size):
            cur_imgs = imgs[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_imgs))
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                self.embeddings.append(embedding_out)
            self.backend.infer_async()
        self.num_features = len(cur_imgs)

    def postprocess(self):
        if self.num_features == 0:
            return np.empty((0, self.feature_dim))

        embedding_out = self.backend.synchronize()[0][:self.num_features * self.feature_dim]
        self.embeddings.append(embedding_out)
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        self._normalize(img, idx, self.backend.input.host, self.input_size)

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img, idx, out, size):
        offset = idx * size
        # BGR to RGB
        img = img[..., ::-1]
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        img = img * (1 / 255)
        img[0, ...] = (img[0, ...] - 0.485) / 0.229
        img[1, ...] = (img[1, ...] - 0.456) / 0.224
        img[2, ...] = (img[2, ...] - 0.406) / 0.225
        # return img.ravel()
        out[offset:offset + size] = img.ravel()

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