from pathlib import Path
import json

from cython_bbox import bbox_overlaps
import numpy as np
import numba as nb
import math
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


def draw_det(frame, det, idx):
    # def lsb_pos(n):
    #     return int(math.log2(n & -n))
    
    def get_tile_id(n):
        tile_id = math.log2(n)
        return int(tile_id) if tile_id.is_integer() else -1
    tlbr = det.tlbr.astype(int)
    tl, br = tlbr[:2], tlbr[2:]
    # tile_id = lsb_pos(det.tile_id)
    tile_id = det.tile_id
    # tile_id = get_tile_id(det.tile_id)
    # text = "%.2f" % det.conf
    text = "%d %.2f" % (idx, det.conf)
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.rectangle(frame, tuple(tl), tuple(br), COLORS[tile_id], 2)
    cv2.rectangle(frame, tuple(tl), (tl[0] + text_width - 1, tl[1] - text_height + 1), COLORS[tile_id], cv2.FILLED)
    cv2.putText(frame, text, tuple(tl), cv2.FONT_HERSHEY_DUPLEX, 0.7, 0, 1, cv2.LINE_AA)


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
        self.label_mask = np.zeros(len(COCO_LABELS), dtype=bool)
        self.label_mask[classes] = True

        self.batch_size = ObjectDetector.config['batch_size']
        self.tile_overlap = ObjectDetector.config['tile_overlap']
        self.tiling_grid = ObjectDetector.config['tiling_grid']
        self.conf_thresh = ObjectDetector.config['conf_thresh']
        self.max_area = ObjectDetector.config['max_area']
        self.merge_iou_thresh = ObjectDetector.config['merge_iou_thresh']

        assert self.batch_size == np.prod(self.tiling_grid) or self.batch_size == 1

        self.model = SSDInceptionV2
        self.input_size = np.prod(self.model.INPUT_SHAPE)
        self.tiles, self.tiling_region_size = self._generate_tiles()

        self.scale_factor = np.asarray(self.size) / self.tiling_region_size
        self.cur_id = -1

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
        tic = time.perf_counter()
        frame = cv2.resize(frame, self.tiling_region_size)
        # imgs = multi_crop(frame, self.tiles)
        self._preprocess(frame, self.tiles, self.backend.input.host, self.input_size)

        # if self.batch_size > 1:
        #     # for i, tile in enumerate(self.tiles):
        #     #     self._preprocess(i, crop(frame, tile))
        #     for i, img in enumerate(imgs):
        #         self._preprocess(i, img)
        # else:
        #     self.cur_id = (self.cur_id + 1) % len(self.tiles)
        #     self._preprocess(0, crop(frame, self.cur_tile))

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

        detections, areas = self._filter_dets(det_out, self.tiles, self.model.TOPK, 
            self.model.OUTPUT_LAYOUT, self.label_mask, self.max_area, self.conf_thresh, self.scale_factor)
        print('loop over det out', time.perf_counter() - tic)

        orig = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        # orig = None

        tic = time.perf_counter()
        detections = self._merge_dets(detections, areas)
        print('merge det', time.perf_counter() - tic)
        return detections, orig

    def draw_tile(self, frame):
        if self.cur_tile is not None:
            cv2.rectangle(frame, tuple(self.cur_tile.tl), tuple(self.cur_tile.br), 0, 1)
        else:
            for tile in self.tiles:
                tl = np.rint(tile[:2] * self.scale_factor).astype(int)
                br = np.rint(tile[2:] * self.scale_factor).astype(int)
                cv2.rectangle(frame, tuple(tl), tuple(br), 0, 1)

    def _generate_tiles(self):
        # frame_size = np.asarray(self.size)
        tile_size, tiling_grid = np.asarray(self.model.INPUT_SHAPE[:0:-1]), np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = tuple(total_size.astype(int))
        # assert all(total_size <= frame_size), "Frame size too small for %dx%d tiles" % self.tiling_grid
        # offset = (size - total_size) // 2
        # tiles = [Rect(tlwh=(c * step_size[0] + offset[0], r * step_size[1] + offset[1], *tile_size)) 
        #     for r in range(tiling_grid[1]) for c in range(tiling_grid[0])]
        # tiles = [Rect(tlwh=(c * step_size[0], r * step_size[1], *tile_size)) 
        #     for r in range(tiling_grid[1]) for c in range(tiling_grid[0])]

        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size)) 
            for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, total_size

    # def _preprocess(self, idx, img):
    #     img = self._normalize(img)
    #     self.backend.memcpy(img, idx)

    def _merge_dets(self, detections, areas):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        if len(detections) == 0:
            return detections

        # merge detections across different tiles
        # bboxes = np.ascontiguousarray([det.bbox.tlbr for det in detections], dtype=np.float)

        # sort by pairwise IOU
        # detections = detections[np.argsort(areas)[::-1]]
        bboxes = detections.tlbr
        ious = bbox_overlaps(bboxes, bboxes)
        sorted_idx = ious.argsort()[:, ::-1]

        # if len(detections) > 61 and detections[61].tile_id == 1 << 6 and abs(detections[61].conf - 0.49) < 0.1:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ious[61, idx[61]], idx[61])

        detections = self._merge(detections, ious, sorted_idx, self.merge_iou_thresh)
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

    # @staticmethod
    # @nb.njit(fastmath=True, cache=True)
    # def _normalize(img):
    #     # BGR to RGB
    #     img = img[..., ::-1]
    #     # HWC -> CHW
    #     img = img.transpose(2, 0, 1)
    #     # Normalize to [-1.0, 1.0] interval (expected by model)
    #     img = img * (2 / 255) - 1
    #     return img.ravel()
    
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
        areas = []
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
                if label_mask[label]:
                    tl = (det_out[offset + 3:offset + 5] * size + tile[:2]) * scale_factor
                    br = (det_out[offset + 5:offset + 7] * size + tile[:2]) * scale_factor
                    tlbr = as_rect(np.append(tl, br))
                    _area = area(tlbr)
                    if _area <= max_area:
                        # np.append(detections, [tlbr, label, conf, 1 << tile_idx])
                        detections.append((tlbr, label, conf, tile_idx))
                        areas.append(_area)
        # detections = np.asarray(detections, dtype=DET_DTYPE)
        return detections, areas

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _merge(dets, ious, sorted_idx, thresh):
        # group adjacent detections
        neighbors = [[i for i in range(0)] for _ in range(len(dets))]
        for i in nb.prange(len(dets)):
            det = dets[i]
            _neighbors = dict()
            for j in range(len(dets)):
                other = dets[j]
                if det.tile_id != other.tile_id and det.label == other.label:
                    if contains(det.tlbr, other.tlbr) or contains(other.tlbr, det.tlbr) or ious[i, j] >= thresh:
                        if _neighbors.get(other.tile_id) is None or ious[i, j] > ious[i, _neighbors[other.tile_id]]:
                            _neighbors[other.tile_id] = j
            neighbors[i].extend(_neighbors.values())
        
        # merge with depth-first traversal
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            candidates = []
            if len(neighbors[i]) > 0 and dets[i].tile_id != -1:
                dets[i].tile_id = -1
                stack.append(i)
                while len(stack) > 0:
                    for j in neighbors[stack.pop()].values():
                        if dets[j].tile_id != -1:
                            candidates.append(j)
                            dets[j].tile_id = -1
                            stack.append(j)
                for k in candidates:
                    dets[i].tlbr[:] = union(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        keep = np.asarray(list(keep))
        

        # keep = set(range(len(dets)))
        # for i in range(len(dets)):
        #     if i in keep:
        #         for j in sorted_idx[i]:
        #             if j in keep:
        #                 if dets[i].tile_id & dets[j].tile_id == 0 and dets[i].label == dets[j].label:
        #                     # if j == 52 and dets[j].tile_id == 1 << 6:
        #                     #     print(dets[j].conf, ious[i, j])
        #                     if contains(dets[i].tlbr, dets[j].tlbr) or ious[i, j] > thresh:
        #                         dets[i].tlbr[:] = union(dets[i].tlbr, dets[j].tlbr)
        #                         dets[i].conf = max(dets[i].conf, dets[j].conf)
        #                         dets[i].tile_id |= dets[j].tile_id
        #                         keep.discard(j)
        # keep = np.asarray(list(keep))
        return dets[keep]
