import numpy as np
import numba as nb
import cv2

from .utils import InferenceBackend
from .models import *


class FeatureExtractor:
    def __init__(self):
        # initialize parameters
        self.model = OSNet025
        self.batch_size = 32
        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.metric = self.model.METRIC
        self.backend = InferenceBackend(self.model, self.batch_size)

    def __call__(self, frame, detections):
        self.preprocess(frame, detections)
        self.backend.infer_async_v2()
        return self.postprocess()

    # def encode(self, frame, detections):
    #     self.encode_async(frame, detections)
    #     return self.postprocess()

    def extract_async(self, frame, detections):
        self.preprocess(frame, detections)
        self.backend.infer_async_v2()
    
    def preprocess(self, frame, detections):
        self.num_detections = len(detections)
        assert self.num_detections <= self.batch_size
        for i, det in enumerate(detections):
            roi = det.bbox.crop(frame)
            roi = cv2.resize(roi, self.model.INPUT_SHAPE[:0:-1])
            self.backend.memcpy(self._preprocess(roi), i)

    def postprocess(self):
        embedding_out = self.backend.synchronize()[0]
        embeddings = [embedding_out[i:i + self.feature_dim] for i in 
            range(0, self.num_detections * self.feature_dim, self.feature_dim)]
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _preprocess(img):
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


# def preproc(img):
#     img = cv2.resize(img, (128, 256))
#     img = img[..., ::-1]
#     img = img.transpose(2, 0, 1)

#     img = img * (1 / 255)
#     img[0, ...] = (img[0, ...] - 0.485) / 0.229
#     img[1, ...] = (img[1, ...] - 0.456) / 0.224
#     img[2, ...] = (img[2, ...] - 0.406) / 0.225

#     # img = img * (2 / 255) - 1
#     return img


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

# imgs = []
# for i, f in enumerate(os.listdir('../test/')):
#     print(i, f)
#     imgs.append(cv2.imread('../test/' + f))


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



# backend = InferenceBackend(OSNet025, 32)

# for i, img in enumerate(imgs):
#     img = preproc(img)
#     backend.memcpy(img.ravel(), i)

# backend.infer_async_v2()
# out = backend.synchronize()[0]

# features = [out[i:i + 512] for i in range(0, len(imgs) * 512, 512)]
# features /= np.linalg.norm(features, axis=1, keepdims=True)

# # print(out2)
# from scipy.spatial.distance import cdist
# # print(cdist(out1[None, ...], out2[None, ...], 'euclidean'))

# print(cdist(features, features, 'euclidean'))