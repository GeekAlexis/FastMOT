import numpy as np
import cv2


class Rect:
    def __init__(self, tlbr=None, tlwh=None):
        if tlbr is not None:
            self.xmin, self.ymin, self.xmax, self.ymax = tlbr
            self.size = (self.xmax - self.xmin + 1, self.ymax - self.ymin + 1)
        elif tlwh is not None:
            self.xmin, self.ymin = tlwh[:2]
            self.size = tuple(tlwh[2:])
            self.xmax = self.size[0] - 1 + self.xmin
            self.ymax = self.size[1] - 1 + self.ymin
        else:
            raise ValueError('Either tlbr or tlwh must not be None') 
        
    def __repr__(self):
        return "Rect(tlbr=(%r, %r, %r, %r))" % (self.xmin, self.ymin, self.xmax, self.ymax)

    def __contains__(self, point):
        return point[0] >= self.xmin and point[1] >= self.ymin and point[0] <= self.xmax and point[1] <= self.ymax

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

    def contains_rect(self, other):
        return other.xmin >= self.xmin and other.ymin >= self.ymin and other.xmax <= self.xmax and other.ymax <= self.ymax

    def tlbr(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def tlwh(self):
        return (self.xmin, self.ymin, self.size[0], self.size[1])
    
    def tlbr_wh(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax, self.size[0], self.size[1])

    def tl(self):
        return (self.xmin, self.ymin)

    def br(self):
        return (self.xmax, self.ymax)

    def center(self):
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def corners(self):
        return (self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax)

    def area(self):
        return self.size[0] * self.size[1]

    def crop(self, image):
        return image[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def scale(self, sx, sy):
        half_size = (self.size * np.array([sx, sy]) - 1) / 2
        xmin, ymin = np.int_(np.round(self.center() - half_size))
        xmax, ymax = np.int_(np.round(self.center() + half_size))
        return Rect(tlbr=(xmin, ymin, xmax, ymax))

    def resize(self, size):
        dx, dy = (np.asarray(size) - self.size) / 2
        xmin = int(round(self.xmin - dx))
        ymin = int(round(self.ymin - dy))
        return Rect(tlwh=(xmin, ymin, *size))
    
    def warp(self, m):
        warped_corners = perspectiveTransform(self.corners(), m)
        return boundingRect(warped_corners)

    def iou(self, other):
        overlap_xmin = max(self.xmin, other.xmin) 
        overlap_ymin = max(self.ymin, other.ymin)
        overlap_xmax = min(self.xmax, other.xmax)
        overlap_ymax = min(self.ymax, other.ymax)
        area_intersection = max(0, overlap_xmax - overlap_xmin + 1) * max(0, overlap_ymax - overlap_ymin + 1)
        return area_intersection / (self.area() + other.area() - area_intersection)


def transform(pts, m):
    pts = np.atleast_2d(pts)
    pts = np.insert(pts, pts.shape[1], 1, axis=1).T
    return (m @ pts).T


def perspectiveTransform(pts, m):
    pts = np.atleast_2d(pts)
    pts = np.insert(pts, pts.shape[1], 1, axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


def boundingRect(pts):
    xmin, ymin = np.int_(np.round(np.min(pts, axis=0)))
    xmax, ymax = np.int_(np.round(np.max(pts, axis=0)))
    return Rect(tlbr=(xmin, ymin, xmax, ymax))