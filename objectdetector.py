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
        cv2.rectangle(frame, self.bbox.tl(), self.bbox.br(), (127, 255, 0), 2)
        cv2.rectangle(frame, self.bbox.tl(), (self.bbox.xmin + text_width - 1, self.bbox.ymin - text_height + 1), (127, 255, 0), cv2.FILLED)
        cv2.putText(frame, text, self.bbox.tl(), cv2.FONT_HERSHEY_SIMPLEX, 1, (143, 48, 0), 2, cv2.LINE_AA)


class ObjectDetector:
    def __init__(self, size, classes=set(range(len(coco_labels))), conf_threshold=0.6, schedule_tiles=True):
        # initialize parameters
        self.size = size
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.max_det = 20
        self.batch_size = 1
        self.tile_overlap = 0.25
        self.tile_size = model.input_shape[1:][::-1]
        self.tiles = self._generate_tiles(self.size, self.tile_size, (3, 2), self.tile_overlap)
        self.tile_aging = np.zeros(len(self.tiles))
        self.schedule_tiles = schedule_tiles

        # initialize TensorRT
        ctypes.CDLL("lib/libflattenconcat.so")
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # load model and create engine
        with open(model.path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        assert self.max_det <= model.topk
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
        self.cur_tile = -1
    
    def preprocess(self, frame, tracks={}, acquire=True):
        if acquire:
            # tile scheduling
            if self.schedule_tiles:
                sx = sy = 1 - self.tile_overlap
                tile_num_tracks = np.zeros(len(self.tiles))
                for tile_id, tile in enumerate(self.tiles):
                    scaled_tile = tile.scale(sx, sy)
                    for track in tracks.values():
                        if track.bbox.center() in scaled_tile or tile.contains_rect(track.bbox):
                            tile_num_tracks[tile_id] += 1
                max_aging = max(self.tile_aging)
                max_num_tracks = max(tile_num_tracks)
                tile_scores = self.tile_aging * (0.6 / (max_aging if max_aging > 0 else 1)) + tile_num_tracks * (0.4 / (max_num_tracks if max_num_tracks > 0 else 1))
                self.cur_tile = np.argmax(tile_scores)
                self.tile_aging += 1
                self.tile_aging[self.cur_tile] = 0
            else:
                self.cur_tile = (self.cur_tile + 1) % len(self.tiles)
            self.roi = self.tiles[self.cur_tile]
        else:
            if len(tracks) > 0:
                track = next(iter(tracks.values()))
                xmin, ymin = np.int_(np.round(track.bbox.center() - (np.array(self.tile_size) - 1) / 2))
                xmin = min(xmin, self.size[0] - self.tile_size[0])
                ymin = min(ymin, self.size[1] - self.tile_size[1])
                self.roi = Rect(cv_rect=(xmin, ymin, self.tile_size[0], self.tile_size[1]))

        tile = self.roi.crop(frame)
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
        # print('[INFO] Latency: %.3f' % (time.time() - tic))

    def postprocess(self):
        self.stream.synchronize()
        output = self.host_outputs[0]
        detections = []
        for det_idx in range(self.max_det):
            offset = det_idx * model.output_layout
            # index = int(output[offset])
            label = int(output[offset + 1])
            conf  = output[offset + 2]
            xmin  = int(round(output[offset + 3] * self.roi.size[0])) + self.roi.xmin
            ymin  = int(round(output[offset + 4] * self.roi.size[1])) + self.roi.ymin
            xmax  = int(round(output[offset + 5] * self.roi.size[0])) + self.roi.xmin
            ymax  = int(round(output[offset + 6] * self.roi.size[1])) + self.roi.ymin
            if conf > self.conf_threshold and label in self.classes:
                bbox = Rect(tf_rect=(xmin, ymin, xmax, ymax))
                det = Detection(bbox, label, conf)
                detections.append(det)
                print('[Detector] Detected: %s' % det)

        # TODO: NMS
        # print('[Detector] Latency: %.3f' % (time.time() - tic))
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

    def detect_sync(self, frame, tracks={}, acquire=True):
        self.preprocess(frame, tracks, acquire)
        self.infer_async()
        return self.postprocess()

    def _generate_tiles(self, size, tile_size, tiling_grid, overlap):
        # TODO: dynamic tiling
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
