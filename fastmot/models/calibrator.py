import os
import numpy as np
import cupy as cp
import tensorrt as trt
import cv2


class SSDEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, model_shape, data_dir, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.model_shape = model_shape
        self.num_calib_imgs = 100 # the number of images from the dataset to use for calibration
        self.batch_size = 10
        self.batch_shape = (self.batch_size, *self.model_shape)
        self.cache_file = cache_file

        calib_imgs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.calib_imgs = np.random.choice(calib_imgs, self.num_calib_imgs)
        self.counter = 0 # for keeping track of how many files we have read

        self.input_dev = cp.empty(self.batch_shape, dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):

        # if there are not enough calibration images to form a batch,
        # we have reached the end of our data set
        if self.counter == self.num_calib_imgs:
            return None

        batch_imgs = np.zeros((self.batch_size, trt.volume(self.model_shape)))
        for i in range(self.batch_size):
            img = cv2.imread(self.calib_imgs[self.counter + i])
            img = cv2.resize(img, (self.model_shape[2], self.model_shape[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # HWC -> CHW
            img = img.transpose((2, 0, 1))
            # Normalize to [-1.0, 1.0] interval (expected by model)
            img = (2.0 / 255.0) * img - 1.0
            # add this image to the batch array
            batch_imgs[i, :] = img.ravel()

        # increase the counter for this batch
        self.counter += self.batch_size

        # Copy to device, then return a list containing pointers to input device buffers.
        batch_imgs = batch_imgs.astype(np.float32)
        self.input_dev.data.copy_from_host(batch_imgs.ctypes.data, batch_imgs.nbytes)
        return [self.input_dev.data.ptr]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
