from collections import defaultdict
from pathlib import Path
import configparser
import abc
import numpy as np
import numba as nb
import cupy as cp
import cupyx.scipy.ndimage
import cv2

from . import models
from .utils import TRTInference
from .utils.rect import as_rect, to_tlbr, get_size, area
from .utils.rect import union, multi_crop, iom, diou_nms


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class Detector(abc.ABC):
    @abc.abstractmethod
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        self.detect_async(frame)
        return self.postprocess()

    @abc.abstractmethod
    def detect_async(self, frame):
        """Detects objects asynchronously."""
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self):
        """Synchronizes, applies postprocessing, and returns a record array
        of detections (DET_DTYPE).
        This function should be called after `detect_async`.
        """
        raise NotImplementedError


class SSDDetector(Detector):
    def __init__(self, size,
                 model='SSDInceptionV2',
                 class_ids=None,
                 tile_overlap=0.25,
                 tiling_grid=(4, 2),
                 conf_thresh=0.5,
                 max_area=130000,
                 merge_thresh=0.6):
        super().__init__(size)
        self.model = getattr(models, model)
        assert 0 <= tile_overlap <= 1
        self.tile_overlap = tile_overlap
        assert tiling_grid[0] >= 1 and tiling_grid[1] >= 1
        self.tiling_grid = tiling_grid
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert max_area >= 0
        self.max_area = max_area
        assert 0 <= merge_thresh <= 1
        self.merge_thresh = merge_thresh

        class_ids = [] if class_ids is None else list(class_ids)
        self.label_mask = np.zeros(len(models.LABEL_MAP), dtype=bool)
        self.label_mask[class_ids] = True

        self.batch_size = int(np.prod(self.tiling_grid))
        self.tiles, self.tiling_region_sz = self._generate_tiles()
        self.scale_factor = np.asarray(self.size) / self.tiling_region_sz
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)

    def detect_async(self, frame):
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self):
        det_out = self.backend.synchronize()[0]
        detections, tile_ids = self._filter_dets(det_out, self.tiles, self.model.TOPK,
                                                 self.label_mask, self.max_area,
                                                 self.conf_thresh, self.scale_factor)
        detections = self._merge_dets(detections, tile_ids)
        return detections

    def _preprocess(self, frame):
        frame = cv2.resize(frame, self.tiling_region_sz)
        self._normalize(frame, self.tiles, self.inp_handle)

    def _generate_tiles(self):
        tile_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])
        tiling_grid = np.asarray(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = np.rint(total_size).astype(int)
        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size))
                          for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, tuple(total_size)

    def _merge_dets(self, detections, tile_ids):
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        tile_ids = np.asarray(tile_ids)
        if len(detections) == 0:
            return detections
        detections = self._merge(detections, tile_ids, self.batch_size, self.merge_thresh)
        return detections.view(np.recarray)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _normalize(frame, tiles, out):
        imgs = multi_crop(frame, tiles)
        for i in nb.prange(len(imgs)):
            bgr = imgs[i]
            # BGR to RGB
            rgb = bgr[..., ::-1]
            # HWC -> CHW
            chw = rgb.transpose(2, 0, 1)
            # Normalize to [-1.0, 1.0] interval
            out[i] = chw * (2 / 255.) - 1.

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
                    if 0 < area(tlbr) <= max_area:
                        detections.append((tlbr, label, conf))
                        tile_ids.append(tile_idx)
        return detections, tile_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, tile_ids, num_tile, thresh):
        # find duplicate neighbors across tiles
        neighbors = [[0 for _ in range(0)] for _ in range(len(dets))]
        for i, det in enumerate(dets):
            max_ioms = np.zeros(num_tile)
            for j, other in enumerate(dets):
                if tile_ids[i] != tile_ids[j] and det.label == other.label:
                    overlap = iom(det.tlbr, other.tlbr)
                    # use the detection with the greatest IoM from each tile
                    if overlap >= thresh and overlap > max_ioms[tile_ids[j]]:
                        max_ioms[tile_ids[j]] = overlap
                        neighbors[i].append(j)

        # merge neighbors using depth-first search
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            if len(neighbors[i]) > 0 and tile_ids[i] != -1:
                tile_ids[i] = -1
                stack.append(i)
                candidates = []
                while len(stack) > 0:
                    for j in neighbors[stack.pop()]:
                        if tile_ids[j] != -1:
                            candidates.append(j)
                            tile_ids[j] = -1
                            stack.append(j)
                for k in candidates:
                    dets[i].tlbr[:] = union(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        keep = np.asarray(list(keep))
        return dets[keep]


class YOLODetector(Detector):
    def __init__(self, size,
                 model='YOLOv4',
                 class_ids=None,
                 conf_thresh=0.25,
                 max_area=800000,
                 nms_thresh=0.5):
        super().__init__(size)
        self.model = getattr(models, model)
        self.class_ids = tuple() if class_ids is None else class_ids
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert max_area >= 0
        self.max_area = max_area
        assert 0 <= nms_thresh <= 1
        self.nms_thresh = nms_thresh

        self.backend = TRTInference(self.model, 1)
        self.inp_handle, self.upscaled_sz, self.bbox_offset = self._create_letterbox()

    def detect_async(self, frame):
        self._preprocess(frame)
        self.backend.infer_async(from_device=True)

    def postprocess(self):
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)
        detections = self._filter_dets(det_out, self.upscaled_sz, self.class_ids, self.conf_thresh,
                                       self.nms_thresh, self.max_area, self.bbox_offset)
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        return detections

    def _preprocess(self, frame):
        zoom = np.roll(self.inp_handle.shape, -1) / frame.shape
        with self.backend.stream:
            frame_dev = cp.asarray(frame)
            # resize
            small_dev = cupyx.scipy.ndimage.zoom(frame_dev, zoom, order=1, mode='opencv', grid_mode=True)
            # BGR to RGB
            rgb_dev = small_dev[..., ::-1]
            # HWC -> CHW
            chw_dev = rgb_dev.transpose(2, 0, 1)
            # normalize to [0, 1] interval
            cp.multiply(chw_dev, 1 / 255., out=self.inp_handle)

    def _create_letterbox(self):
        src_size = np.asarray(self.size)
        dst_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])
        if self.model.LETTERBOX:
            scale_factor = min(dst_size / src_size)
            scaled_size = np.rint(src_size * scale_factor).astype(int)
            img_offset = (dst_size - scaled_size) / 2
            roi = np.s_[:, img_offset[1]:img_offset[1] + scaled_size[1],
                        img_offset[0]:img_offset[0] + scaled_size[0]]
            upscaled_sz = np.rint(dst_size / scale_factor).astype(int)
            bbox_offset = (upscaled_sz - src_size) / 2
        else:
            roi = np.s_[:]
            upscaled_sz = src_size
            bbox_offset = np.zeros(2)
        inp_reshaped = self.backend.input.device.reshape(self.model.INPUT_SHAPE)
        inp_reshaped[:] = 0.5 # initial value for letterbox
        inp_handle = inp_reshaped[roi]
        return inp_handle, upscaled_sz, bbox_offset

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, size, class_ids, conf_thresh, nms_thresh, max_area, offset):
        """
        det_out: a list of 3 tensors, where each tensor
                 contains a multiple of 7 float32 numbers in
                 the order of [x, y, w, h, box_confidence, class_id, class_prob]
        """
        # drop detections with low score
        scores = det_out[:, 4] * det_out[:, 6]
        keep = np.where(scores >= conf_thresh)
        det_out = det_out[keep]

        # scale to pixel values
        det_out[:, :4] *= np.append(size, size)
        det_out[:, :2] -= offset

        keep = []
        for class_id in class_ids:
            class_idx = np.where(det_out[:, 5] == class_id)[0]
            class_dets = det_out[class_idx]
            class_keep = diou_nms(class_dets[:, :4], class_dets[:, 4], nms_thresh)
            keep.extend(class_idx[class_keep])
        keep = np.asarray(keep)
        nms_dets = det_out[keep]

        detections = []
        for i in range(len(nms_dets)):
            tlbr = to_tlbr(nms_dets[i, :4])
            label = int(nms_dets[i, 5])
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if 0 < area(tlbr) <= max_area:
                detections.append((tlbr, label, conf))
        return detections


class PublicDetector(Detector):
    def __init__(self, size, frame_skip, sequence_path=None, conf_thresh=0.5, max_area=800000):
        super().__init__(size)
        self.frame_skip = frame_skip
        assert sequence_path is not None
        self.seq_root = Path(__file__).parents[1] / sequence_path
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert max_area >= 0
        self.max_area = max_area

        assert self.seq_root.exists()
        seqinfo = configparser.ConfigParser()
        seqinfo.read(self.seq_root / 'seqinfo.ini')
        self.seq_size = (int(seqinfo['Sequence']['imWidth']), int(seqinfo['Sequence']['imHeight']))

        self.detections = defaultdict(list)
        self.frame_id = 0

        det_txt = self.seq_root / 'det' / 'det.txt'
        for mot_challenge_det in np.loadtxt(det_txt, delimiter=','):
            frame_id = int(mot_challenge_det[0]) - 1
            tlbr = to_tlbr(mot_challenge_det[2:6])
            conf = 1.0 # mot_challenge_det[6]
            label = 1 # mot_challenge_det[7] (person)
            # scale inside frame
            tlbr[:2] = tlbr[:2] / self.seq_size * self.size
            tlbr[2:] = tlbr[2:] / self.seq_size * self.size
            tlbr = as_rect(tlbr)
            if conf >= self.conf_thresh and area(tlbr) <= self.max_area:
                self.detections[frame_id].append((tlbr, label, conf))

    def detect_async(self, frame):
        pass

    def postprocess(self):
        detections = np.asarray(self.detections[self.frame_id], dtype=DET_DTYPE).view(np.recarray)
        self.frame_id += self.frame_skip
        return detections
