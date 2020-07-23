from pathlib import Path
import json
from copy import deepcopy
import numpy as np
import numba as nb
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
        return "Detection(bbox=%r, label=%r, conf=%r, tile_id=%r)" % (self.bbox, self.label, 
            self.conf, self.tile_id)

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
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['ObjectDetector']

    def __init__(self, size, classes):
        # initialize parameters
        self.size = size
        self.classes = set(classes)
        self.max_det = ObjectDetector.config['max_det']
        self.batch_size = ObjectDetector.config['batch_size']
        self.tile_overlap = ObjectDetector.config['tile_overlap']
        self.tiling_grid = ObjectDetector.config['tiling_grid']
        self.conf_thresh = ObjectDetector.config['conf_thresh']
        self.max_area = ObjectDetector.config['max_area']
        self.merge_iou_thresh = ObjectDetector.config['merge_iou_thresh']

        self.model = SSDInceptionV2
        self.tile_max_det = self.max_det // np.prod(self.tiling_grid)
        self.tile_size = self.model.INPUT_SHAPE[:0:-1]
        self.tiles = self._generate_tiles()
        self.cur_id = -1

        assert self.batch_size == np.prod(self.tiling_grid) or self.batch_size == 1
        assert self.tile_max_det <= self.model.TOPK

        self.backend = InferenceBackend(self.model, self.batch_size)

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    @property
    def tiling_region(self):
        return Rect(tlbr=(*self.tiles[0].tl, *self.tiles[-1].br))

    @property
    def cur_tile(self):
        if self.cur_id == -1:
            return None
        return self.tiles[self.cur_id]

    def detect_async(self, frame):
        # TODO: resize frame to tiling region?
        tic = time.perf_counter()
        if self.batch_size > 1:
            for i, tile in enumerate(self.tiles):
                self._preprocess(i, tile.crop(frame))
        else:
            self.cur_id = (self.cur_id + 1) % len(self.tiles)
            self._preprocess(0, self.cur_tile.crop(frame))

        print('img pre', time.perf_counter() - tic)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        # print(time.perf_counter() - self.tic)

        tic = time.perf_counter()
        detections = []
        for tile_id in range(self.batch_size):
            tile = self.tiles[tile_id] if self.batch_size > 1 else self.cur_tile
            tile_offset = tile_id * self.model.TOPK
            for det_idx in range(self.tile_max_det):
                offset = (tile_offset + det_idx) * self.model.OUTPUT_LAYOUT
                # index = int(det_out[offset])
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf > self.conf_thresh and label in self.classes:
                    tl = det_out[offset + 3:offset + 5] * tile.size + tile.tl
                    br = det_out[offset + 5:offset + 7] * tile.size + tile.tl
                    bbox = Rect(tlbr=(*tl, *br))
                    # TODO: filter out large detection?
                    # TODO: split duplicate long detection?
                    if bbox.area <= self.max_area:
                        detections.append(Detection(bbox, label, conf, tile_id))
                    # print('[Detector] Detected: %s' % det)
        print('loop over det out', time.perf_counter() - tic)
        orig_dets = deepcopy(detections)

        tic = time.perf_counter()
        # merge detections across different tiles
        keep = set(range(len(detections)))
        for i, det in enumerate(detections):
            if i in keep:
                for j, other in enumerate(detections):
                        if other.tile_id.isdisjoint(det.tile_id) and det.label == other.label:
                            if other.bbox in det.bbox or det.bbox.iou(other.bbox) > self.merge_iou_thresh:
                                det.bbox |= other.bbox
                                det.tile_id |= other.tile_id
                                det.conf = max(det.conf, other.conf)
                                keep.discard(j)
        detections = np.asarray(detections)
        detections = detections[list(keep)]

        print('merge det', time.perf_counter() - tic)
        return detections, orig_dets

    def draw_tile(self, frame):
        if self.cur_tile is not None:
            cv2.rectangle(frame, tuple(self.cur_tile.tl), tuple(self.cur_tile.br), 0, 2)
        else:
            [cv2.rectangle(frame, tuple(tile.tl), tuple(tile.br), 0, 2) for tile in self.tiles]

    def _generate_tiles(self):
        size = np.asarray(self.size)
        tile_size, tiling_grid = np.asarray(self.tile_size), np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        assert all(total_size <= size), "Frame size too small for %dx%d tiles" % self.tiling_grid
        offset = (size - total_size) // 2
        tiles = [Rect(tlwh=(c * step_size[0] + offset[0], r * step_size[1] + offset[1], *tile_size)) 
            for r in range(tiling_grid[1]) for c in range(tiling_grid[0])]
        return tiles

    def _preprocess(self, idx, img):
        img = self._normalize(img)
        self.backend.memcpy(img, idx)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _normalize(img):
        # BGR to RGB
        img = img[..., ::-1]
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img = img * (2 / 255) - 1
        return img.ravel()
        