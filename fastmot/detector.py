from pathlib import Path
import logging
import json

from cython_bbox import bbox_overlaps
import numpy as np
import numba as nb
from numba.typed import Dict
import ctypes
import cv2
import time

from .models import *
from .utils import ConfigDecoder, InferenceBackend
from .utils.rect import *


DET_DTYPE = np.dtype([
    ('tlbr', float, 4), 
    ('label', int), 
    ('conf', float), 
    ('tile_id', int)], 
    align=True
)


# DET_DTYPE = np.dtype([
#     ('tlbr', float, 4), 
#     ('label', int), 
#     ('conf', float)], 
#     align=True
# )


class ObjectDetector:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['ObjectDetector']

    def __init__(self, size, class_ids):
        # initialize parameters
        self.size = size
        self.label_mask = np.zeros(len(COCO_LABELS), dtype=bool)
        self.label_mask[class_ids] = True

        self.tile_overlap = ObjectDetector.config['tile_overlap']
        self.tiling_grid = ObjectDetector.config['tiling_grid']
        self.conf_thresh = ObjectDetector.config['conf_thresh']
        self.max_area = ObjectDetector.config['max_area']
        self.merge_iou_thresh = ObjectDetector.config['merge_iou_thresh']
        self.batch_size = int(np.prod(self.tiling_grid))

        self.model = SSDInceptionV2
        self.input_size = np.prod(self.model.INPUT_SHAPE)
        self.tiles, self.tiling_region_size = self._generate_tiles()
        self.scale_factor = np.asarray(self.size) / self.tiling_region_size
        self.backend = InferenceBackend(self.model, self.batch_size)

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    def detect_async(self, frame):
        tic = time.perf_counter()
        frame = cv2.resize(frame, self.tiling_region_size)
        self._preprocess(frame, self.tiles, self.backend.input.host, self.input_size)

        logging.debug('img pre %f', time.perf_counter() - tic)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]

        tic = time.perf_counter()
        detections = self._filter_dets(det_out, self.tiles, self.model.TOPK, 
            self.model.OUTPUT_LAYOUT, self.label_mask, self.max_area, self.conf_thresh, self.scale_factor)
        logging.debug('filter dets %f', time.perf_counter() - tic)

        tic = time.perf_counter()
        detections = self._merge_dets(detections)
        logging.debug('merge dets %f', time.perf_counter() - tic)
        return detections

    def _generate_tiles(self):
        tile_size, tiling_grid = np.asarray(self.model.INPUT_SHAPE[:0:-1]), np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = tuple(total_size.astype(int))
        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size)) 
            for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, total_size

    def _merge_dets(self, detections):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        if len(detections) == 0:
            return detections

        # merge detections across different tiles
        bboxes = detections.tlbr
        ious = bbox_overlaps(bboxes, bboxes)

        detections = self._merge(detections, ious, self.merge_iou_thresh)
        return detections.view(np.recarray)
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _preprocess(frame, tiles, out, size):
        imgs = multi_crop(frame, tiles)
        for i in nb.prange(len(imgs)):
            offset = i * size
            bgr = imgs[i]
            # BGR to RGB
            rgb = bgr[..., ::-1]
            # HWC -> CHW
            chw = rgb.transpose(2, 0, 1)
            # Normalize to [-1.0, 1.0] interval
            normalized = chw * (2 / 255) - 1
            out[offset:offset + size] = normalized.ravel()

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, tiles, topk, layout, label_mask, max_area, thresh, scale_factor):
        detections = []
        # tile_ids = []
        for tile_idx in range(len(tiles)):
            tile = tiles[tile_idx]
            size = get_size(tile)
            tile_offset = tile_idx * topk
            for det_idx in range(topk):
                offset = (tile_offset + det_idx) * layout
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf < thresh:
                    break
                if label_mask[label]:
                    tl = (det_out[offset + 3:offset + 5] * size + tile[:2]) * scale_factor
                    br = (det_out[offset + 5:offset + 7] * size + tile[:2]) * scale_factor
                    tlbr = as_rect(np.append(tl, br))
                    if area(tlbr) <= max_area:
                        detections.append((tlbr, label, conf, tile_idx))
                        # tile_ids.append(tile_idx)
        return detections

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, ious, thresh):
        # find adjacent detections
        neighbors = [Dict.empty(nb.types.int64, nb.types.int64) for _ in range(len(dets))]
        for i in range(len(dets)):
            det = dets[i]
            for j in range(len(dets)):
                other = dets[j]
                if other.tile_id != det.tile_id and other.label == det.label:
                    if contains(det.tlbr, other.tlbr) or contains(other.tlbr, det.tlbr) or ious[i, j] >= thresh:
                        # pick the nearest detection from each tile
                        if neighbors[i].get(other.tile_id) is None or ious[i, j] > ious[i, neighbors[i][other.tile_id]]:
                            neighbors[i][other.tile_id] = j
        
        # merge detections using depth-first search
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            if len(neighbors[i]) > 0 and dets[i].tile_id != -1:
                dets[i].tile_id = -1
                stack.append(i)
                candidates = []
                while len(stack) > 0:
                    for j in neighbors[stack.pop()].values():
                        if dets[j].tile_id != -1:
                            candidates.append(j)
                            dets[j].tile_id = -1
                            stack.append(j)
                # merge candidates
                for k in candidates:
                    dets[i].tlbr[:] = union(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        keep = np.asarray(list(keep))
        return dets[keep]


class YoloDetector:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['YoloDetector']

    try:
        ctypes.cdll.LoadLibrary(Path(__file__).parent / 'plugins' / 'libyolo_layer.so')
    except OSError as err:
        raise RuntimeError('ERROR: failed to load libyolo_layer.so.  '
                        'Did you forget to do a "make" in the "plugins" '
                        'subdirectory?') from err

    def __init__(self, size, class_ids):
        # initialize parameters
        self.size = size
        self.class_ids = coco2yolo(class_ids)

        self.conf_thresh = YoloDetector.config['conf_thresh']
        self.max_area = YoloDetector.config['max_area']
        self.nms_thresh = YoloDetector.config['nms_thresh']
        
        self.model = YoloV4
        self.batch_size = 1
        self.backend = InferenceBackend(self.model, self.batch_size)

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    def detect_async(self, frame):
        tic = time.perf_counter()
        frame = cv2.resize(frame, self.model.INPUT_SHAPE[:0:-1])
        self._preprocess(frame, self.backend.input.host)

        logging.debug('img pre %f', time.perf_counter() - tic)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)

        tic = time.perf_counter()
        detections = self._filter_dets(det_out, self.size, self.class_ids, self.conf_thresh, self.nms_thresh, self.max_area)
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        logging.debug('filter dets %f', time.perf_counter() - tic)
        return detections
    
    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _preprocess(frame, out):
        # BGR to RGB
        rgb = frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize to [0, 1] interval
        normalized = chw / 255.
        out[:] = normalized.ravel()

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, size, class_ids, conf_thresh, nms_thresh, max_area):
        """
        det_out: a list of 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        """
        size = np.asarray(size)

        # drop detections with score lower than conf_thresh
        scores = det_out[:, 4] * det_out[:, 6]
        keep = np.where(scores >= conf_thresh)[0]
        det_out = det_out[keep]

        # scale to pixel values
        det_out[:, :2] *= size
        det_out[:, 2:4] *= size

        keep = []
        for class_id in class_ids:
            class_idx = np.where(det_out[:, 5] == class_id)[0]
            class_dets = det_out[class_idx]
            class_keep = nms(class_dets[:, :4], class_dets[:, 4], nms_thresh)
            keep.extend(class_idx[class_keep])
        keep = np.asarray(keep)
        nms_dets = det_out[keep]
        
        detections = []
        for i in range(len(nms_dets)):
            tlbr = to_tlbr(nms_dets[i, :4])
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(size, size))
            label = YOLO2COCO[int(nms_dets[i, 5])]
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if area(tlbr) <= max_area:
                detections.append((tlbr, label, conf, -1))
        return detections
