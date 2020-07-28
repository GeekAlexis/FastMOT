from pathlib import Path
import json

from cython_bbox import bbox_overlaps
import numpy as np
import numba as nb
import cv2
import time

from .utils import *
from .models import *


COLORS = [
    (75, 25, 230), 
    (48, 130, 245), 
    (25, 225, 255), 
    (60, 245, 210), 
    (75, 180, 60), 
    (240, 240, 70), 
    (200, 130, 0), 
    (180, 30, 145), 
    (230, 50, 240)
 ]


DET_DTYPE = np.dtype([
    ('tlbr', float, 4), 
    ('label', int), 
    ('conf', float), 
    ('tile_id', int)], 
    align=True
)


class Detection:
    def __init__(self, bbox, label, conf, tile_id):
        self.bbox = bbox
        self.label = label
        self.conf = conf
        self.tile_id = set([tile_id])

    def __repr__(self):
        return "Detection(bbox=%r, label=%r, conf=%r, tile_id=%r)" % (self.bbox, self.label, 
            self.conf, self.tile_id)

    def __str__(self):
        return "%.2f %s at %s" % (self.conf, COCO_LABELS[self.label], self.bbox.tlwh)
    
    def draw(self, frame):
        # text = "%s: %.2f" % (COCO_LABELS[self.label], self.conf) 
        text = "%.2f" % self.conf
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(frame, tuple(self.bbox.tl), tuple(self.bbox.br), COLORS[next(iter(self.tile_id))], 2)
        cv2.rectangle(frame, tuple(self.bbox.tl), (self.bbox.xmin + text_width - 1, 
            self.bbox.ymin - text_height + 1), COLORS[next(iter(self.tile_id))], cv2.FILLED)
        cv2.putText(frame, text, tuple(self.bbox.tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, 0, 1, cv2.LINE_AA)


class ObjectDetector:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['ObjectDetector']

    def __init__(self, size, classes):
        # initialize parameters
        self.size = size
        self.classes = set(classes)
        self.batch_size = ObjectDetector.config['batch_size']
        self.tile_overlap = ObjectDetector.config['tile_overlap']
        self.tiling_grid = ObjectDetector.config['tiling_grid']
        self.conf_thresh = ObjectDetector.config['conf_thresh']
        self.max_area = ObjectDetector.config['max_area']
        self.merge_iou_thresh = ObjectDetector.config['merge_iou_thresh']

        self.model = SSDInceptionV2
        self.tiles = self._generate_tiles()
        # self.tiling_region = Rect(tlbr=(*self.tiles[0].tl, *self.tiles[-1].br))

        tiling_region = np.r_[self.tiles[0][:2], self.tiles[-1][2:]]
        self.tiling_region_size = tuple(get_size(tiling_region).astype(int))
        self.scale_factor = np.asarray(self.size) / self.tiling_region_size
        self.cur_id = -1

        assert self.batch_size == np.prod(self.tiling_grid) or self.batch_size == 1

        self.backend = InferenceBackend(self.model, self.batch_size)

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    @property
    def cur_tile(self):
        if self.cur_id == -1:
            return None
        return self.tiles[self.cur_id]

    def detect_async(self, frame):
        # TODO: resize frame to tiling region?
        tic = time.perf_counter()
        frame = cv2.resize(frame, self.tiling_region_size)
        if self.batch_size > 1:
            for i, tile in enumerate(self.tiles):
                self._preprocess(i, crop(frame, tile))
        else:
            self.cur_id = (self.cur_id + 1) % len(self.tiles)
            self._preprocess(0, crop(frame, self.cur_tile))

        print('img pre', time.perf_counter() - tic)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        # print(time.perf_counter() - self.tic)

        tic = time.perf_counter()
        # detections = []
        # for tile_idx in range(self.batch_size):
        #     tile = self.tiles[tile_id] #if self.batch_size > 1 else self.cur_tile
        #     tile_offset = tile_idx * self.model.TOPK
        #     for det_idx in range(self.tile_max_det):
        #         offset = (tile_offset + det_idx) * self.model.OUTPUT_LAYOUT
        #         # index = int(det_out[offset])
        #         label = int(det_out[offset + 1])
        #         conf = det_out[offset + 2]
        #         if conf > self.conf_thresh and label in self.classes:
        #             tl = (det_out[offset + 3:offset + 5] * tile.size + tile.tl) * self.scale_factor
        #             br = (det_out[offset + 5:offset + 7] * tile.size + tile.tl) * self.scale_factor
        #             # bbox = Rect(tlbr=(*tl, *br))
        #             tlbr = as_rect((*tl, *br))
        #             # TODO: rec array, duplicate long det?
        #             # if bbox.area <= self.max_area:
        #             if area(tlbr) <= self.max_area:
        #                 detections.append((tlbr, label, conf, 1 << tile_idx))
        #                 # detections.append(Detection(bbox, label, conf, tile_id))
        #             # print('[Detector] Detected: %s' % det)

        detections = self._filter_detections(det_out, self.tiles, self.model.TOPK, 
            self.model.OUTPUT_LAYOUT, self.scale_factor, self.classes, self.conf_thresh, self.max_area)
        print('loop over det out', time.perf_counter() - tic)

        # orig_dets = detections

        tic = time.perf_counter()
        detections = self._merge_detections(detections)
        print('merge det', time.perf_counter() - tic)
        return detections

    def draw_tile(self, frame):
        if self.cur_tile is not None:
            cv2.rectangle(frame, tuple(self.cur_tile.tl), tuple(self.cur_tile.br), 0, 1)
        else:
            [cv2.rectangle(frame, tuple(tile.tl), tuple(tile.br), 0, 1) for tile in self.tiles]

    def _generate_tiles(self):
        size = np.asarray(self.size)
        tile_size, tiling_grid = np.asarray(self.model.INPUT_SHAPE[:0:-1]), np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        assert all(total_size <= size), "Frame size too small for %dx%d tiles" % self.tiling_grid
        # offset = (size - total_size) // 2
        # tiles = [Rect(tlwh=(c * step_size[0] + offset[0], r * step_size[1] + offset[1], *tile_size)) 
        #     for r in range(tiling_grid[1]) for c in range(tiling_grid[0])]
        # tiles = [Rect(tlwh=(c * step_size[0], r * step_size[1], *tile_size)) 
        #     for r in range(tiling_grid[1]) for c in range(tiling_grid[0])]

        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size)) 
            for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
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

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_detections(det_out, tiles, topk, layout, scale_factor, classes, thresh, max_area):
        detections = []
        # detections = np.empty(0, dtype=DET_DTYPE)
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
                if label in classes:
                    tl = (det_out[offset + 3:offset + 5] * size + tile[:2]) * scale_factor
                    br = (det_out[offset + 5:offset + 7] * size + tile[:2]) * scale_factor
                    tlbr = as_rect(np.append(tl, br))
                    if area(tlbr) <= max_area:
                        # np.append(detections, [tlbr, label, conf, 1 << tile_idx])
                        detections.append((tlbr, label, conf, 1 << tile_idx))
        # detections = np.asarray(detections, dtype=DET_DTYPE)
        return detections

    def _merge_detections(self, detections):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        if len(detections) == 0:
            return detections

        # merge detections across different tiles
        # bboxes = np.ascontiguousarray([det.bbox.tlbr for det in detections], dtype=np.float)
        bboxes = detections.tlbr
        ious = bbox_overlaps(bboxes, bboxes)
        idx = ious.argsort()[:, :0:-1]
        detections = self._merge(detections, ious, idx, self.merge_iou_thresh)
        return detections.view(np.recarray)

        # keep = set(range(len(detections)))
        # dirty = set()
        # for i, det in enumerate(detections):
        #     if i in keep:
        #         for j, other in zip(idx[i], detections[idx[i]]):
        #             if j in keep and j not in dirty:
        #                 if other.tile_id.isdisjoint(det.tile_id) and det.label == other.label:
        #                     if other.bbox in det.bbox or ious[i, j] > self.merge_iou_thresh:
        #                         det.bbox |= other.bbox
        #                         det.tile_id |= other.tile_id
        #                         det.conf = max(det.conf, other.conf)
        #                         keep.discard(j)
        #                         dirty.add(i)
        # detections = detections[list(keep)]
        
    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, ious, sorted_idx, thresh):
        keep = set(range(len(dets)))
        for i in range(len(dets)):
            if i in keep:
                for j in sorted_idx[i]:
                    if j in keep:
                        if dets[i].tile_id & dets[j].tile_id == 0 and dets[i].label == dets[j].label:
                            if contains(dets[i].tlbr, dets[j].tlbr) or ious[i, j] > thresh:
                                dets[i].tlbr[:] = union(dets[i].tlbr, dets[j].tlbr)
                                dets[i].conf = max(dets[i].conf, dets[j].conf)
                                dets[i].tile_id |= dets[j].tile_id
                                keep.discard(j)
        keep = np.asarray(list(keep))
        return dets[keep]
