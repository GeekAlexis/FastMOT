import numpy as np
import numba as nb
import cv2


class Rect:
    def __init__(self, xmin, ymin, width, height):
        self.xmin = int(round(xmin))
        self.ymin = int(round(ymin))
        self.size = (int(round(width)), int(round(height)))
        self.xmax = self.size[0] - 1 + self.xmin
        self.ymax = self.size[1] - 1 + self.ymin
        
    def __repr__(self):
        return "Rect(%r, %r, %r, %r)" % (self.xmin, self.ymin, self.size[0], self.size[1])

    @property
    def tlbr(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])
    
    @property
    def tlwh(self):
        return np.array([self.xmin, self.ymin, self.size[0], self.size[1]])
    
    @property
    def tlbr_wh(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax, 
            self.size[0], self.size[1]])

    @property
    def tl(self):
        return np.array([self.xmin, self.ymin])

    @property
    def br(self):
        return np.array([self.xmax, self.ymax])

    @property
    def center(self):
        return np.array([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2])

    @property
    def corners(self):
        return np.array([[self.xmin, self.ymin], [self.xmax, self.ymin], 
            [self.xmax, self.ymax], [self.xmin, self.ymax]])

    @property
    def area(self):
        return self.size[0] * self.size[1]

    def __and__(self, other):
        # intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        w, h = xmax - xmin + 1, ymax - ymin + 1
        if w <= 0 or h <= 0:
            return None
        return Rect(xmin, ymin, w, h)

    def __or__(self, other):
        # minimum rect that contains both rects
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        w, h = xmax - xmin + 1, ymax - ymin + 1
        return Rect(xmin, ymin, w, h)

    def contains_rect(self, other):
        return other.xmin >= self.xmin and other.ymin >= self.ymin and \
            other.xmax <= self.xmax and other.ymax <= self.ymax

    def crop(self, image):
        return image[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def scale(self, sx, sy):
        size = np.asarray(self.size) * np.array([sx, sy])
        xmin, ymin = self.center - (size - 1) / 2
        return Rect(xmin, ymin, *size)

    def resize(self, size):
        dx, dy = (np.asarray(size) - np.asarray(self.size)) / 2
        xmin, ymin = self.xmin - dx, self.ymin - dy
        return Rect(xmin, ymin, *size)
    
    def warp(self, m):
        warped_corners = perspectiveTransform(self.corners, m)
        tl = np.min(warped_corners, axis=0)
        br = np.max(warped_corners, axis=0)
        w, h = br - tl + 1
        return Rect(*tl, w, h)

    def iou(self, other):
        overlap_xmin = max(self.xmin, other.xmin) 
        overlap_ymin = max(self.ymin, other.ymin)
        overlap_xmax = min(self.xmax, other.xmax)
        overlap_ymax = min(self.ymax, other.ymax)
        area_intersection = max(0, overlap_xmax - overlap_xmin + 1) * max(0, 
            overlap_ymax - overlap_ymin + 1)
        return area_intersection / (self.area + other.area - area_intersection)


@nb.njit(fastmath=True, cache=True)
def transform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    return (m @ pts).T


@nb.njit(fastmath=True, cache=True)
def perspectiveTransform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


@nb.njit(parallel=True, fastmath=True, cache=True)
def iou(bbox, candidates):
    """Vectorized version of intersection over union.
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, bottom right x, bottom right y)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    bbox, candidates = np.asarray(bbox), np.asarray(candidates)
    if len(candidates) == 0:
        return np.zeros(len(candidates))

    area_bbox = np.prod(bbox[2:] - bbox[:2] + 1)
    area_candidates = np.prod(candidates[:, 2:] - candidates[:, :2] + 1, axis=1)

    overlap_xmin = np.maximum(bbox[0], candidates[:, 0])
    overlap_ymin = np.maximum(bbox[1], candidates[:, 1])
    overlap_xmax = np.minimum(bbox[2], candidates[:, 2])
    overlap_ymax = np.minimum(bbox[3], candidates[:, 3])
    
    area_intersection = np.maximum(0, overlap_xmax - overlap_xmin + 1) * \
                        np.maximum(0, overlap_ymax - overlap_ymin + 1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)