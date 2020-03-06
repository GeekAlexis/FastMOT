from enum import Enum
import cv2

from .objectdetector import ObjectDetector
from .kalmantracker import KalmanTracker


class Analytics:
    class Status(Enum):
        SEARCHING, TARGET_NOT_FOUND, TARGET_ACQUIRED, TARGET_LOST = (i for i in range(4))

    def __init__(self, size, capture_dt, enable_drawing=False):
        self.size = size
        self.enable_drawing = enable_drawing
        self.acq_detector_frame_skip = 3
        self.trk_detector_frame_skip = 5
        self.acquisition_interval = 100
        # self.classes = set([1]) # person only
        self.classes = set([1, 2, 3, 22, 24]) # person, bicycle, car, elephant, zebra

        ObjectDetector.init_backend()
        print('[Analytics] Loading acquisition detector model...')
        self.acq_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.ACQUISITION)
        print('[Analytics] Loading tracking detector model...')
        self.trk_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.TRACKING)
        self.tracker = KalmanTracker(self.size, capture_dt)
        
        # reset flags
        self.status = Analytics.Status.SEARCHING
        self.acquire = True
        self.detector = self.acq_detector
        self.detector_frame_skip = self.acq_detector_frame_skip
        self.acquisition_start_frame = 0
        self.track_id = -1
        self.frame_count = 0
    
    def run(self, frame):
        detections = []
        if self.frame_count == 0:
            detections = self.detector.detect_sync(frame)
            self.tracker.init(frame, detections)
            print('\n[Analytics] Acquiring new targets...')
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.preprocess(frame, self.tracker.tracks, track_id=self.track_id)
                self.detector.infer_async()
                self.tracker.track(frame)
                detections = self.detector.postprocess()
                self.tracker.update(detections, self.detector.cur_tile, self.detector.tile_overlap, acquire=self.acquire)
            else:
                self.tracker.track(frame)

        if self.acquire:
            if self.frame_count - self.acquisition_start_frame == self.acquisition_interval:
                if len(self.tracker.tracks) > 0:
                    self.track_id = self.tracker.get_nearest_track()
                    self.acquire = False
                    self.detector = self.trk_detector
                    self.detector_frame_skip = self.trk_detector_frame_skip
                    print('[Analytics] Following: %s' % self.tracker.tracks[self.track_id])
                    self.status = Analytics.Status.TARGET_ACQUIRED
                else:
                    self.acquisition_start_frame = self.frame_count
                    self.status = Analytics.Status.TARGET_NOT_FOUND
            else:
                self.status = Analytics.Status.SEARCHING
        elif self.track_id not in self.tracker.tracks:
            self.acquire = True
            self.detector = self.acq_detector
            self.detector_frame_skip = self.acq_detector_frame_skip
            self.acquisition_start_frame = self.frame_count
            print('[Analytics] Acquiring new targets...')
            self.status = Analytics.Status.TARGET_LOST

        if self.enable_drawing:
            self._draw(frame, detections)
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_cur_tile(frame)

        self.frame_count += 1

    def reset(self):
        self.acquire = True
        self.status = Analytics.Status.SEARCHING
        self.detector = self.acq_detector
        self.detector_frame_skip = self.acq_detector_frame_skip
        self.acquisition_start_frame = 0
        self.frame_count = 0

    def get_target_bbox(self):
        assert self.status == Analytics.Status.TARGET_ACQUIRED
        return self.tracker.tracks[self.track_id].bbox

    def _draw(self, frame, detections):
        for track_id, track in self.tracker.tracks.items():
            if self.status == Analytics.Status.TARGET_ACQUIRED and track_id == self.track_id:
                track.draw(frame, follow=True, draw_feature_match=True)
            else:
                track.draw(frame, draw_feature_match=True)
        [det.draw(frame) for det in detections]
        if self.acquire:
            cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        elif self.status == Analytics.Status.TARGET_ACQUIRED:
            cv2.putText(frame, 'Following %d' % self.track_id, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    