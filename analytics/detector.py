from enum import Enum
from pathlib import Path
import json

import numpy as np
import cv2
import time

from .utils import *
from .models import *


class Detection:
    def __init__(self, bbox, label, conf, tile_id):
        self.bbox = bbox
        self.label = label
        self.conf = conf
        if isinstance(tile_id, int):
            self.tile_id = set([tile_id])
        else:
            self.tile_id = tile_id

    def __repr__(self):
        return "Detection(bbox=%r, label=%r, conf=%r, tile_id=%r)" % (self.bbox, self.label, self.conf, self.tile_id)

    def __str__(self):
        return "%.2f %s at %s" % (self.conf, COCO_LABELS[self.label], self.bbox.tlwh)
    
    def draw(self, frame):
        text = "%s: %.2f" % (COCO_LABELS[self.label], self.conf) 
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), (112, 25, 25), 2)
        cv2.rectangle(frame, tuple(self.bbox.tl), (self.bbox.xmin + text_width - 1, 
            self.bbox.ymin - text_height + 1), (112, 25, 25), cv2.FILLED)
        cv2.putText(frame, text, tuple(self.bbox.tl), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 255), 
            2, cv2.LINE_AA)


class ObjectDetector:
    class Type(Enum):
        TRACKING = 0
        ACQUISITION = 1

    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['ObjectDetector']

    def __init__(self, size, classes, detector_type):
        # initialize parameters
        self.size = size
        self.classes = set(classes)
        self.detector_type = detector_type
        self.max_det = ObjectDetector.config['max_det']
        self.batch_size = ObjectDetector.config['batch_size']
        self.tile_overlap = ObjectDetector.config['tile_overlap']
        self.merge_iou_thresh = ObjectDetector.config['merge_iou_thresh']

        self.tiles = None
        self.cur_tile = None
        self.cur_tile_id = -1
        if self.detector_type == ObjectDetector.Type.ACQUISITION:
            self.tiling_grid = ObjectDetector.config['acquisition']['tiling_grid']
            self.conf_threshold = ObjectDetector.config['acquisition']['conf_threshold']
            self.model = SSDInceptionV2 #SSDMobileNetV1
            self.tile_size = self.model.INPUT_SHAPE[:0:-1]
            self.tiles = self._generate_tiles()
            assert self.batch_size == np.prod(self.tiling_grid) or self.batch_size == 1
        elif self.detector_type == ObjectDetector.Type.TRACKING:
            self.conf_threshold = ObjectDetector.config['tracking']['conf_threshold']
            self.model = SSDInceptionV2
            self.tile_size = self.model.INPUT_SHAPE[:0:-1]
            assert self.batch_size == 1, 'Only batch size = 1 is supported for tracking detector'
        else:
            raise ValueError(f'Invalid detector type; must be either {ObjectDetector.Type.ACQUISITION} or \
                             {ObjectDetector.Type.TRACKING}')
        assert self.max_det <= self.model.TOPK
        self.backend = InferenceBackend(self.model, self.batch_size)
        self.input_batch = np.empty((self.batch_size, np.prod(self.model.INPUT_SHAPE)))
    
    @property
    def tiling_region(self):
        assert self.tiles is not None
        size = self.tiles[-1].br - self.tiles[0].tl + 1
        return Rect(*self.tiles[0].tl, *size)

    def preprocess(self, frame, roi=None):
        if self.detector_type == ObjectDetector.Type.ACQUISITION:
            if self.batch_size > 1:
                for i, tile in enumerate(self.tiles):
                    frame_tile = tile.crop(frame)
                    frame_tile = cv2.cvtColor(frame_tile, cv2.COLOR_BGR2RGB)
                    frame_tile = np.transpose(frame_tile, (2, 0, 1)) # HWC -> CHW
                    self.input_batch[i] = frame_tile.ravel()
                # Normalize to [-1.0, 1.0] interval (expected by model)
                self.input_batch = self.input_batch * (2 / 255) - 1
            else:
                self.cur_tile_id = (self.cur_tile_id + 1) % len(self.tiles)
                self.cur_tile = self.tiles[self.cur_tile_id]
        elif self.detector_type == ObjectDetector.Type.TRACKING:
            tile_size = np.asarray(self.tile_size)
            tl = roi.center - (tile_size - 1) / 2
            tl = np.clip(tl, 0, self.size - tile_size)
            self.cur_tile = Rect(*tl, *self.tile_size)

        if self.cur_tile is not None:
            frame_tile = self.cur_tile.crop(frame)
            frame_tile = cv2.cvtColor(frame_tile, cv2.COLOR_BGR2RGB)
            frame_tile = np.transpose(frame_tile, (2, 0, 1))
            frame_tile = frame_tile * (2 / 255) - 1
            self.input_batch[0] = frame_tile.ravel()
        return self.input_batch

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        # print(time.perf_counter() - self.tic)
        detections = []
        for tile_idx in range(self.batch_size):
            tile = self.tiles[tile_idx] if self.batch_size > 1 else self.cur_tile
            tile_offset = tile_idx * self.model.TOPK
            for det_idx in range(self.max_det):
                offset = (tile_offset + det_idx) * self.model.OUTPUT_LAYOUT
                # index = int(det_out[offset])
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf > self.conf_threshold and label in self.classes:
                    # xmin = int(round(det_out[offset + 3] * tile.size[0])) + tile.xmin
                    # ymin = int(round(det_out[offset + 4] * tile.size[1])) + tile.ymin
                    # xmax = int(round(det_out[offset + 5] * tile.size[0])) + tile.xmin
                    # ymax = int(round(det_out[offset + 6] * tile.size[1])) + tile.ymin
                    xmin = det_out[offset + 3] * tile.size[0] + tile.xmin
                    ymin = det_out[offset + 4] * tile.size[1] + tile.ymin
                    xmax = det_out[offset + 5] * tile.size[0] + tile.xmin
                    ymax = det_out[offset + 6] * tile.size[1] + tile.ymin
                    w, h = xmax - xmin + 1, ymax - ymin + 1
                    bbox = Rect(xmin, ymin, w, h)
                    detections.append(Detection(bbox, label, conf, tile_idx))
                    # print('[Detector] Detected: %s' % det)

        # merge detections across different tiles
        merged_detections = []
        merged_det_indices = set()
        for i, det1 in enumerate(detections):
            if i not in merged_det_indices:
                merged_det = Detection(det1.bbox, det1.label, det1.conf, det1.tile_id)
                for j, det2 in enumerate(detections):
                    if j not in merged_det_indices:
                        if not det2.tile_id.issubset(merged_det.tile_id) and merged_det.label == det2.label:
                            if merged_det.bbox.contains_rect(det2.bbox) or merged_det.bbox.iou(det2.bbox) > self.merge_iou_thresh:
                                merged_det.bbox |= det2.bbox
                                merged_det.conf = max(merged_det.conf, det2.conf) 
                                merged_det.tile_id |= det2.tile_id
                                merged_det_indices.add(i)
                                merged_det_indices.add(j)
                if i in merged_det_indices:
                    merged_detections.append(merged_det)
        detections = np.delete(detections, list(merged_det_indices))
        detections = np.r_[detections, merged_detections]
        return detections

    def detect(self, frame, roi=None):
        self.detect_async(frame, roi)
        return self.postprocess()

    def detect_async(self, frame, roi=None):
        inp = self.preprocess(frame, roi)
        self.backend.infer_async(inp)

    def draw_tile(self, frame):
        if self.cur_tile is not None:
            cv2.rectangle(frame, tuple(self.cur_tile.tl), tuple(self.cur_tile.br), 0, 2)
        else:
            [cv2.rectangle(frame, tuple(tile.tl), tuple(tile.br), 0, 2) for tile in self.tiles]

    def _generate_tiles(self):
        width, height = self.size
        tile_width, tile_height = self.tile_size
        step_width = (1 - self.tile_overlap) * tile_width
        step_height = (1 - self.tile_overlap) * tile_height
        total_width = (self.tiling_grid[0] - 1) * step_width + tile_width
        total_height = (self.tiling_grid[1] - 1) * step_height + tile_height
        assert total_width <= width and total_height <= height, "Frame size not large enough for %dx%d tiles" % self.tiling_grid
        x_offset = width // 2 - total_width // 2
        y_offset = height // 2 - total_height // 2
        tiles = [Rect(c * step_width + x_offset, r * step_height + y_offset, tile_width, tile_height) for r in
                range(self.tiling_grid[1]) for c in range(self.tiling_grid[0])]
        return tiles
        
