from pathlib import Path
import configparser

from cython_bbox import bbox_overlaps
from collections import defaultdict
from numba.typed import Dict
import numpy as np
import numba as nb
import cv2

from . import models
from .utils import InferenceBackend
from .utils.rect import *


DET_DTYPE = np.dtype([
    ('tlbr', float, 4), 
    ('label', int), 
    ('conf', float)], 
    align=True
)


class Detector:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_id, frame):
        self.detect_async(frame_id, frame)
        return self.postprocess()

    def detect_async(self, frame_id, frame):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError


class SSDDetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.label_mask = np.zeros(len(models.LABEL_MAP), dtype=bool)
        self.label_mask[config['class_ids']] = True

        self.model = getattr(models, config['model'])
        self.tile_overlap = config['tile_overlap']
        self.tiling_grid = config['tiling_grid']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.merge_thresh = config['merge_thresh']

        self.batch_size = int(np.prod(self.tiling_grid))
        self.input_size = np.prod(self.model.INPUT_SHAPE)
        self.tiles, self.tiling_region_size = self._generate_tiles()
        self.scale_factor = np.asarray(self.size) / self.tiling_region_size
        self.backend = InferenceBackend(self.model, self.batch_size)

    def detect_async(self, frame_id, frame):
        frame = cv2.resize(frame, self.tiling_region_size)
        self._preprocess(frame, self.tiles, self.backend.input.host, self.input_size)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        detections, tile_ids = self._filter_dets(det_out, self.tiles, self.model.TOPK, 
            self.label_mask, self.max_area, self.conf_thresh, self.scale_factor)
        detections = self._merge_dets(detections, tile_ids)
        return detections

    def _generate_tiles(self):
        tile_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])
        tiling_grid = np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = tuple(total_size.astype(int))
        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size)) 
            for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, total_size

    def _merge_dets(self, detections, tile_ids):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        tile_ids = np.asarray(tile_ids)
        if len(detections) == 0:
            return detections

        # merge detections across different tiles
        bboxes = detections.tlbr
        ious = bbox_overlaps(bboxes, bboxes)
        detections = self._merge(detections, tile_ids, ious, self.merge_thresh)
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
    def _filter_dets(det_out, tiles, topk, label_mask, max_area, thresh, scale_factor):
        detections = []
        tile_ids = []
        for tile_idx in range(len(tiles)):
            tile = tiles[tile_idx]
            size = get_size(tile)
            tile_offset = tile_idx * topk
            for det_idx in range(topk):
                offset = (tile_offset + det_idx) * 7
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf < thresh:
                    break
                if label_mask[label]:
                    tl = (det_out[offset + 3:offset + 5] * size + tile[:2]) * scale_factor
                    br = (det_out[offset + 5:offset + 7] * size + tile[:2]) * scale_factor
                    tlbr = as_rect(np.append(tl, br))
                    if area(tlbr) <= max_area:
                        detections.append((tlbr, label, conf))
                        tile_ids.append(tile_idx)
        return detections, tile_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, tile_ids, ious, thresh):
        # find adjacent detections
        neighbors = [Dict.empty(nb.types.int64, nb.types.int64) for _ in range(len(dets))]
        for i in range(len(dets)):
            cur_neighbors = neighbors[i]
            for j in range(len(dets)):
                if tile_ids[j] != tile_ids[i] and dets[i].label == dets[j].label:
                    if contains(dets[i].tlbr, dets[j].tlbr) or contains(dets[j].tlbr, dets[i].tlbr) or \
                        ious[i, j] >= thresh:
                        # pick the nearest detection from each tile
                        if cur_neighbors.get(tile_ids[j]) is None or ious[i, j] > ious[i, cur_neighbors[tile_ids[j]]]:
                            cur_neighbors[tile_ids[j]] = j
        
        # merge detections using depth-first search
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            if len(neighbors[i]) > 0 and tile_ids[i] != -1:
                tile_ids[i] = -1
                stack.append(i)
                candidates = []
                while len(stack) > 0:
                    for j in neighbors[stack.pop()].values():
                        if tile_ids[j] != -1:
                            candidates.append(j)
                            tile_ids[j] = -1
                            stack.append(j)
                # merge candidates
                for k in candidates:
                    dets[i].tlbr[:] = union(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        keep = np.asarray(list(keep))
        return dets[keep]


class YoloDetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.model = getattr(models, config['model'])
        self.class_ids = config['class_ids']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.nms_thresh = config['nms_thresh']
        
        self.batch_size = 1
        self.backend = InferenceBackend(self.model, self.batch_size)

    def detect_async(self, frame_id, frame):
        frame = cv2.resize(frame, self.model.INPUT_SHAPE[:0:-1])
        self._preprocess(frame, self.backend.input.host)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)
        detections = self._filter_dets(det_out, self.size, self.class_ids, self.conf_thresh, 
            self.nms_thresh, self.max_area)
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
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
            # clip inside frame
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(size, size))
            label = int(nms_dets[i, 5])
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if area(tlbr) <= max_area:
                detections.append((tlbr, label, conf))
        return detections


class PublicDetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.seq_root = Path(__file__).parents[1] / config['sequence']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']

        seqinfo = configparser.ConfigParser()
        seqinfo.read(self.seq_root / 'seqinfo.ini')
        self.seq_size = (int(seqinfo['Sequence']['imWidth']), int(seqinfo['Sequence']['imHeight']))

        self.detections = defaultdict(list)
        self.query_fid = None

        det_txt = self.seq_root / 'det' / 'det.txt'
        for frame_id, _, x, y, w, h, conf in np.loadtxt(det_txt, delimiter=','):
            tlbr = to_tlbr((x, y, w, h))
            # scale and clip inside frame
            tlbr[:2] = tlbr[:2] / self.seq_size * self.size
            tlbr[2:] = tlbr[2:] / self.seq_size * self.size
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(self.size, self.size))
            tlbr = as_rect(tlbr)
            if conf >= self.conf_thresh and area(tlbr) <= self.max_area:
                self.detections[int(frame_id)].append((tlbr, 1, conf))

    def detect_async(self, frame_id, frame):
        self.query_fid = frame_id + 1

    def postprocess(self):
        return np.asarray(self.detections[self.query_fid], dtype=DET_DTYPE).view(np.recarray)
