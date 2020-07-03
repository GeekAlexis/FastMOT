import numpy as np
import numba as nb
import cv2


spec = [
    ('xmin', nb.int32),
    ('xmax', nb.int32),
    ('ymin', nb.int32),
    ('ymax', nb.int32),
    ('size', nb.types.UniTuple(nb.int32, 2))
]


# @nb.jitclass(spec)
class Rect:
    def __init__(self, xmin, ymin, width, height):
        self.xmin = int(round(xmin))
        self.ymin = int(round(ymin))
        self.size = (int(round(width)), int(round(height)))
        self.xmax = self.size[0] - 1 + self.xmin
        self.ymax = self.size[1] - 1 + self.ymin
        
    # def __repr__(self):
    #     return "Rect(tlbr=(%r, %r, %r, %r))" % (self.xmin, self.ymin, self.xmax, self.ymax)

    # def __contains__(self, point):
    #     return point[0] >= self.xmin and point[1] >= self.ymin and point[0] <= self.xmax and point[1] <= self.ymax

    # def __and__(self, other):
    #     # intersection
    #     xmin = max(self.xmin, other.xmin)
    #     ymin = max(self.ymin, other.ymin)
    #     xmax = min(self.xmax, other.xmax)
    #     ymax = min(self.ymax, other.ymax)
    #     intersection = Rect(tlbr=(xmin, ymin, xmax, ymax))
    #     if intersection.size[0] <= 0 or intersection.size[1] <= 0:
    #         return None
    #     return intersection

    # def __or__(self, other):
    #     # minimum rect that contains both rects
    #     xmin = min(self.xmin, other.xmin)
    #     ymin = min(self.ymin, other.ymin)
    #     xmax = max(self.xmax, other.xmax)
    #     ymax = max(self.ymax, other.ymax)
    #     return Rect(tlbr=(xmin, ymin, xmax, ymax))

    def contains(self, point):
        return point[0] >= self.xmin and point[1] >= self.ymin and \
            point[0] <= self.xmax and point[1] <= self.ymax

    def intersect(self, other):
        # intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        w, h = xmax - xmin + 1, ymax - ymin + 1
        intersection = Rect(xmin, ymin, w, h)
        if intersection.size[0] <= 0 or intersection.size[1] <= 0:
            return None
        return intersection

    def union(self, other):
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
        xmin, ymin = min(warped_corners[:, 0]), min(warped_corners[:, 1])
        xmax, ymax = max(warped_corners[:, 0]), max(warped_corners[:, 1])
        w, h = xmax - xmin + 1, ymax - ymin + 1
        return Rect(xmin, ymin, w, h)

    def iou(self, other):
        overlap_xmin = max(self.xmin, other.xmin) 
        overlap_ymin = max(self.ymin, other.ymin)
        overlap_xmax = min(self.xmax, other.xmax)
        overlap_ymax = min(self.ymax, other.ymax)
        area_intersection = max(0, overlap_xmax - overlap_xmin + 1) * max(0, 
            overlap_ymax - overlap_ymin + 1)
        return area_intersection / (self.area + other.area - area_intersection)


@nb.njit(parallel=True, cache=True)
def transform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    # pts = np.insert(pts, pts.shape[1], 1, axis=1).T
    return (m @ pts).T


@nb.njit(parallel=True, cache=True)
def perspectiveTransform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    # pts = np.insert(pts, pts.shape[1], 1, axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


@nb.njit(parallel=True, cache=True)
def iou(bbox, candidates):
    """Vectorized version of intersection over union.
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, bottom right x, bottom right y, width, height)`.
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

    overlap_xmin = np.maximum(bbox[0], candidates[:, 0])
    overlap_ymin = np.maximum(bbox[1], candidates[:, 1])
    overlap_xmax = np.minimum(bbox[2], candidates[:, 2])
    overlap_ymax = np.minimum(bbox[3], candidates[:, 3])
    
    area_intersection = np.maximum(0, overlap_xmax - overlap_xmin + 1) * \
                        np.maximum(0, overlap_ymax - overlap_ymin + 1)
    area_bbox = bbox[4:].prod()
    area_candidates = candidates[:, 4:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)