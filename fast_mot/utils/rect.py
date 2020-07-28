import numpy as np
import numba as nb


class Rect:
    def __init__(self, tlbr=None, tlwh=None):
        if tlbr is not None:
            self.xmin = int(round(tlbr[0]))
            self.ymin = int(round(tlbr[1]))
            self.xmax = int(round(tlbr[2]))
            self.ymax = int(round(tlbr[3]))
            self.size = (self.xmax - self.xmin + 1, self.ymax - self.ymin + 1)
        elif tlwh is not None:
            self.xmin = int(round(tlwh[0]))
            self.ymin = int(round(tlwh[1]))
            self.size = (int(round(tlwh[2])), int(round(tlwh[3])))
            self.xmax = self.xmin + self.size[0] - 1
            self.ymax = self.ymin + self.size[1] - 1
        else:
            raise ValueError('Either tlbr or tlwh must not be None') 
        
    def __repr__(self):
        return "Rect(tlbr=(%r, %r, %r, %r))" % (self.xmin, self.ymin, self.xmax, self.ymax)

    def __contains__(self, other):
        return other.xmin >= self.xmin and other.ymin >= self.ymin and \
            other.xmax <= self.xmax and other.ymax <= self.ymax

    def __and__(self, other):
        # intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        intersection = Rect(tlbr=(xmin, ymin, xmax, ymax))
        if intersection.size[0] <= 0 or intersection.size[1] <= 0:
            return None
        return intersection

    def __or__(self, other):
        # minimum rect that contains both rects
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        return Rect(tlbr=(xmin, ymin, xmax, ymax))

    def __mul__(self, scale):
        # scale rect
        width = self.size[0] * scale[0] - 1
        height = self.size[1] * scale[1] - 1
        center = self.center
        xmin = center[0] - width / 2
        ymin = center[1] - height / 2
        return Rect(tlwh=(xmin, ymin, width, height))

    @property
    def tlbr(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])
    
    @property
    def tlwh(self):
        return np.array([self.xmin, self.ymin, self.size[0], self.size[1]])

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

    def crop(self, img):
        return img[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def resize(self, size):
        dx = (size[0] - self.size[0]) / 2
        dy = (size[1] - self.size[1]) / 2
        xmin = self.xmin - dx
        ymin = self.ymin - dy
        return Rect(tlwh=(xmin, ymin, *size))

    def warp(self, m):
        return Rect(tlbr=self._warp(self.corners, m))

    def iou(self, other):
        overlap_xmin = max(self.xmin, other.xmin) 
        overlap_ymin = max(self.ymin, other.ymin)
        overlap_xmax = min(self.xmax, other.xmax)
        overlap_ymax = min(self.ymax, other.ymax)
        area_intersection = max(0, overlap_xmax - overlap_xmin + 1) * \
            max(0, overlap_ymax - overlap_ymin + 1)
        return area_intersection / (self.area + other.area - area_intersection)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _warp(corners, m):
        warped_corners = perspectiveTransform(corners, m)
        xmin = min(warped_corners[:, 0])
        ymin = min(warped_corners[:, 1])
        xmax = max(warped_corners[:, 0])
        ymax = max(warped_corners[:, 1])
        return xmin, ymin, xmax, ymax


@nb.njit(cache=True)
def as_rect(tlbr):
    tlbr = np.asarray(tlbr, np.float64)
    tlbr = np.rint(tlbr)
    return tlbr


@nb.njit(cache=True)
def get_size(tlbr):
    tl, br = tlbr[:2], tlbr[2:]
    size = br - tl + 1
    return size


@nb.njit(cache=True)
def area(tlbr):
    size = get_size(tlbr)
    return int(size[0] * size[1])


@nb.njit(cache=True)
def get_center(tlbr):
    xmin, ymin, xmax, ymax = tlbr
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])


@nb.njit(cache=True)
def get_corners(tlbr):
    xmin, ymin, xmax, ymax = tlbr
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


@nb.njit(cache=True)
def to_tlwh(tlbr):
    return np.append(tlbr[:2], get_size(tlbr))


@nb.njit(cache=True)
def to_tlbr(tlwh):
    tlwh = np.asarray(tlwh, np.float64)
    tlwh = np.rint(tlwh)
    tl, size = tlwh[:2], tlwh[2:]
    br = tl + size - 1
    return np.append(tl, br)


@nb.njit(cache=True)
def contains(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    return np.all((tl2 >= tl1) & (br2 <= br1))


@nb.njit(cache=True)
def intersection(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.maximum(tl1, tl2)
    br = np.minimum(br1, br2)
    tlbr = np.append(tl, br)
    if np.any(get_size(tlbr) <= 0):
        return None
    return tlbr


@nb.njit(cache=True)
def union(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.minimum(tl1, tl2)
    br = np.maximum(br1, br2)
    tlbr = np.append(tl, br)
    return tlbr


@nb.njit(cache=True)
def crop(img, tlbr):
    xmin, ymin, xmax, ymax = tlbr.astype(np.int_)
    return img[ymin:ymax + 1, xmin:xmax + 1]


@nb.njit(cache=True)
def multi_crop(img, tlbrs):
    _tlbrs = tlbrs.astype(np.int_)
    return [img[_tlbrs[i][1]:_tlbrs[i][3] + 1, _tlbrs[i][0]:_tlbrs[i][2] + 1] for i in range(len(_tlbrs))]


@nb.njit(fastmath=True, cache=True)
def warp(tlbr, m):
    corners = get_corners(tlbr)
    warped_corners = perspectiveTransform(corners, m)
    xmin = min(warped_corners[:, 0])
    ymin = min(warped_corners[:, 1])
    xmax = max(warped_corners[:, 0])
    ymax = max(warped_corners[:, 1])
    return as_rect((xmin, ymin, xmax, ymax))


@nb.njit(fastmath=True, cache=True)
def transform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T


@nb.njit(fastmath=True, cache=True)
def perspectiveTransform(pts, m):
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


# @nb.njit(parallel=True, fastmath=True, cache=True)
# def iou(bbox, candidates):
#     """Vectorized version of intersection over union.
#     Parameters
#     ----------
#     bbox : ndarray
#         A bounding box in format `(top left x, top left y, bottom right x, bottom right y)`.
#     candidates : ndarray
#         A matrix of candidate bounding boxes (one per row) in the same format
#         as `bbox`.
#     Returns
#     -------
#     ndarray
#         The intersection over union in [0, 1] between the `bbox` and each
#         candidate. A higher score means a larger fraction of the `bbox` is
#         occluded by the candidate.
#     """
#     bbox, candidates = np.asarray(bbox), np.asarray(candidates)
#     if len(candidates) == 0:
#         return np.zeros(len(candidates))

#     area_bbox = np.prod(bbox[2:] - bbox[:2] + 1)
#     size_candidates = candidates[:, 2:] - candidates[:, :2] + 1
#     area_candidates = size_candidates[:, 0] * size_candidates[:, 1]

#     overlap_xmin = np.maximum(bbox[0], candidates[:, 0])
#     overlap_ymin = np.maximum(bbox[1], candidates[:, 1])
#     overlap_xmax = np.minimum(bbox[2], candidates[:, 2])
#     overlap_ymax = np.minimum(bbox[3], candidates[:, 3])
    
#     area_intersection = np.maximum(0, overlap_xmax - overlap_xmin + 1) * \
#         np.maximum(0, overlap_ymax - overlap_ymin + 1)
#     return area_intersection / (area_bbox + area_candidates - area_intersection)