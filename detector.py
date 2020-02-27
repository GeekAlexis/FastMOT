import ctypes
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from util import *
import models.ssd as ssd


class DetectorType:
    """
    enumeration type for object detector type
    """
    TRACKING = 0
    ACQUISITION = 1


class Detector:
    runtime = None

    @classmethod
    def init_backend(cls): 
        # initialize TensorRT
        ctypes.CDLL("lib/libflattenconcat.so")
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        ObjectDetector.runtime = trt.Runtime(trt_logger)

    def __init__(self, size, classes=set(range(len(coco_labels))), detector_type=DetectorType.ACQUISITION):
        # initialize parameters
        self.size = size
        self.classes = classes
        self.detector_type = detector_type
        self.max_det = 20
        self.batch_size = 1

        self.tile_overlap = 0.25
        self.tiles = None
        self.cur_tile = None
        if self.detector_type == DetectorType.ACQUISITION:
            self.model = ssd.MobileNetV1
            self.conf_threshold = 0.5
            self.schedule_tiles = True
            self.tile_size = self.model.INPUT_SHAPE[1:][::-1]
            self.tiles = self._generate_tiles(self.size, self.tile_size, (3, 2), self.tile_overlap)
            self.tile_ages = np.zeros(len(self.tiles))
            self.age_to_object_ratio = 0.4
            self.cur_tile_id = -1
        elif self.detector_type == DetectorType.TRACKING:
            self.model = ssd.InceptionV2
            self.conf_threshold = 0.5
            self.tile_size = self.model.INPUT_SHAPE[1:][::-1]

        # load model and create engine
        with open(self.model.PATH, 'rb') as f:
            buf = f.read()
            engine = ObjectDetector.runtime.deserialize_cuda_engine(buf)
        assert self.max_det <= self.model.TOPK
        assert self.batch_size <= engine.max_batch_size

        # create buffers
        self.host_inputs  = []
        self.cuda_inputs  = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * self.batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        self.context = engine.create_execution_context()
        self.input_batch = np.zeros((self.batch_size, trt.volume(self.model.INPUT_SHAPE)))
    
    def preprocess(self, frame, tracks={}, track_id=None):
        if self.detector_type == DetectorType.ACQUISITION:
            # tile scheduling
            if self.schedule_tiles:
                sx = sy = 1 - self.tile_overlap
                tile_num_tracks = np.zeros(len(self.tiles))
                for tile_id, tile in enumerate(self.tiles):
                    scaled_tile = tile.scale(sx, sy)
                    for track in tracks.values():
                        if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox):
                            tile_num_tracks[tile_id] += 1
                tile_scores = self.tile_ages * self.age_to_object_ratio + tile_num_tracks
                self.cur_tile_id = np.argmax(tile_scores)
                self.tile_ages += 1
                self.tile_ages[self.cur_tile_id] = 0
            else:
                self.cur_tile_id = (self.cur_tile_id + 1) % len(self.tiles)
            self.cur_tile = self.tiles[self.cur_tile_id]
        elif self.detector_type == DetectorType.TRACKING:
            assert len(tracks) > 0 and track_id is not None
            xmin, ymin = np.int_(np.round(tracks[track_id].bbox.center() - (np.array(self.tile_size) - 1) / 2))
            xmin = max(min(self.size[0] - self.tile_size[0], xmin), 0)
            ymin = max(min(self.size[1] - self.tile_size[1], ymin), 0)
            self.cur_tile = Rect(cv_rect=(xmin, ymin, self.tile_size[0], self.tile_size[1]))

        tile = self.cur_tile.crop(frame)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile = tile * (2 / 255) - 1 # Normalize to [-1.0, 1.0] interval (expected by model)
        tile = np.transpose(tile, (2, 0, 1)) # HWC -> CHW
        self.input_batch[-1] = tile.ravel()
        np.copyto(self.host_inputs[0], self.input_batch.ravel())

    def infer_async(self):
        # inference
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

    def postprocess(self):
        self.stream.synchronize()
        output = self.host_outputs[0]
        detections = []
        for det_idx in range(self.max_det):
            offset = det_idx * self.model.OUTPUT_LAYOUT
            # index = int(output[offset])
            label = int(output[offset + 1])
            conf = output[offset + 2]
            if conf > self.conf_threshold and label in self.classes:
                xmin = int(round(output[offset + 3] * self.cur_tile.size[0])) + self.cur_tile.xmin
                ymin = int(round(output[offset + 4] * self.cur_tile.size[1])) + self.cur_tile.ymin
                xmax = int(round(output[offset + 5] * self.cur_tile.size[0])) + self.cur_tile.xmin
                ymax = int(round(output[offset + 6] * self.cur_tile.size[1])) + self.cur_tile.ymin
                bbox = Rect(tf_rect=(xmin, ymin, xmax, ymax))
                detections.append(Detection(bbox, label, conf))
                # print('[ObjectDetector] Detected: %s' % det)
        return detections

    def detect_sync(self, frame, tracks={}, track_id=None):
        self.preprocess(frame, tracks, track_id)
        self.infer_async()
        return self.postprocess()

    def get_tiling_region(self):
        assert self.detector_type == DetectorType.ACQUISITION and len(self.tiles) > 0
        return Rect(tf_rect=(self.tiles[0].xmin, self.tiles[0].ymin, self.tiles[-1].xmax, self.tiles[-1].ymax))

    def _generate_tiles(self, size, tile_size, tiling_grid, overlap):
        width, height = size
        tile_width, tile_height = tile_size
        step = 1 - overlap
        total_width = (tiling_grid[0] - 1) * tile_width * step + tile_width
        total_height = (tiling_grid[1] - 1) * tile_height * step + tile_height
        assert total_width <= width and total_height <= height, "Frame size not large enough for %dx%d tiles" % tiling_grid
        x_offset = width // 2 - total_width // 2
        y_offset = height // 2 - total_height // 2
        tiles = [Rect(cv_rect=(int(col * tile_width * step + x_offset), int(row * tile_height * step + y_offset), tile_width, tile_height)) for row in range(tiling_grid[1]) for col in range(tiling_grid[0])]
        return tiles
