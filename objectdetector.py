import numpy as np
import cv2
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from util import *
# import models.ssd_mobilenet_v2 as model
import models.ssd_inception_v2 as model
import time


class Detection:
    def __init__(self, bbox, label, conf):
        self.bbox = bbox
        self.label = label
        self.conf = conf

    def __repr__(self):
        return "Detection(bbox=%r, label=%r, conf=%r)" % (self.bbox, self.label, self.conf)

    def __str__(self):
        return "%s at %s with %.2f confidence" % (coco_labels[self.label], self.bbox.cv_rect(), self.conf)
    
    def draw(self, frame):
        text = "%s: %.2f" % (coco_labels[self.label], self.conf) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), (0, 165, 255), 3)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), (0, 165, 255), cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, (143, 48, 0), 2, cv2.LINE_AA)


class ObjectDetector:
    def __init__(self, size, classes=set(range(len(coco_labels))), conf_threshold=0.6, tiling_grid=(1, 1), max_det=20):
        # initialize parameters
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.batch_size = 1
        self.tile_overlap = 0.25
        self.tiles = self._generate_tiles(size, tiling_grid, self.tile_overlap)
        self.mode = 'acquisition'
        if max_det > model.topk:
            raise Exception('Maximum number of detections must be less than %s' % model.topk)
        # TODO: acquisition vs tracking mode

        # initialize TensorRT
        ctypes.CDLL("lib/libflattenconcat.so")
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # load model and create engine
        with open(model.path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        assert self.batch_size <= engine.max_batch_size

        # create buffer
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
        self.input_batch = np.zeros((self.batch_size, trt.volume(model.input_shape)))
        self.cur_tile = 0
    
    def detect(self, frame):
        # preprocess
        # full = cv2.resize(frame, model.input_shape[1:], interpolation=cv2.INTER_AREA)
        # full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
        # full = full * (2 / 255) - 1 # Normalize to [-1.0, 1.0] interval (expected by model)
        # full = np.transpose(full, (2, 0, 1)) # HWC -> CHW
        # self.input_batch[-1] = full.ravel()
        
        # tic = time.time()
        tile_rect = self.tiles[self.cur_tile]
        tile = tile_rect.crop(frame)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile = tile * (2 / 255) - 1 # Normalize to [-1.0, 1.0] interval (expected by model)
        tile = np.transpose(tile, (2, 0, 1)) # HWC -> CHW
        self.input_batch[-1] = tile.ravel()
        np.copyto(self.host_inputs[0], self.input_batch.ravel())

        # inference
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        # print('[INFO] Latency: %.3f' % (time.time() - tic))

        output = self.host_outputs[0]
        detections = []
        for det_idx in range(self.max_det):
            offset = det_idx * model.output_layout
            # index = int(output[offset])
            label = int(output[offset + 1])
            conf  = output[offset + 2]
            xmin  = int(round(output[offset + 3] * tile_rect.size[0])) + tile_rect.xmin
            ymin  = int(round(output[offset + 4] * tile_rect.size[1])) + tile_rect.ymin
            xmax  = int(round(output[offset + 5] * tile_rect.size[0])) + tile_rect.xmin
            ymax  = int(round(output[offset + 6] * tile_rect.size[1])) + tile_rect.ymin
            if conf > self.conf_threshold and label in self.classes:
                bbox = Rect(tf_rect=(xmin, ymin, xmax, ymax))
                det = Detection(bbox, label, conf)
                detections.append(det)
                print('[Detector] Detected: %s' % det)
        
        # print('[Detector] Latency: %.3f' % (time.time() - tic))

        self.cur_tile = (self.cur_tile + 1) % len(self.tiles)
        # discard = set()
        # for i in range(len(detections)):
        #     if i not in discard:
        #         for j in range(len(detections)):
        #             if j not in discard:
        #                 if detections[i].tile_id != detections[j].tile_id and detections[i].label == detections[j].label and iou(detections[i].bbox, detections[j].bbox) > self.merging_iou_threshold: #and det.conf < other.conf:    
        #                         detections[i].bbox.merge(detection[j].bbox)
        #                         detections[i].conf = max(detections[i].conf, detections[j].conf)
        #                         discard.add(j)
        #     if detections[i].conf < self.conf_threshold:
        #         discard.add(i)
        # merged_detections = np.delete(detections, list(discard))

        return detections

    def get_cur_tile(self, scale_for_overlap=False):
        sx = sy = 1
        if scale_for_overlap:
            sx = sy = 1 - self.tile_overlap
        return self.tiles[self.cur_tile].scale(sx, sy)

    def draw_cur_tile(self, frame):
        self.tiles[self.cur_tile].draw(frame)

    def _generate_tiles(self, size, tiling_grid, overlap=0.25):
        width, height = size
        tile_height, tile_width = model.input_shape[1:]
        step = 1 - overlap
        total_width = (tiling_grid[0] - 1) * tile_width * step + tile_width
        total_height = (tiling_grid[1] - 1) * tile_height * step + tile_height
        assert total_width <= width and total_height <= height, "Frame size not large enough for %dx%d tiles" % tiling_grid
        x_offset = width // 2 - total_width // 2
        y_offset = height // 2 - total_height // 2
        tiles = [Rect(cv_rect=(int(col * tile_width * step + x_offset), int(row * tile_height * step + y_offset), tile_width, tile_height)) for row in range(tiling_grid[1]) for col in range(tiling_grid[0])]
        return tiles
