from enum import Enum
from pathlib import Path
import json
import cv2

from .objectdetector import ObjectDetector
from .kalmantracker import KalmanTracker
from .configs import decoder

class Analytics:
    class Status(Enum):
        SEARCHING, TARGET_NOT_FOUND, TARGET_ACQUIRED, TARGET_LOST = (i for i in range(4))
    
    with open(Path(__file__).parent / 'configs' / 'config.json') as config_file:
        config = json.load(config_file, cls=decoder.decoder)['Analytics']

    def __init__(self, size, capture_dt, enable_drawing=False):
        self.size = size
        self.enable_drawing = enable_drawing
        self.acq_detector_frame_skip = Analytics.config['acq_detector_frame_skip']
        self.trk_detector_frame_skip = Analytics.config['trk_detector_frame_skip']
        self.acquisition_interval = Analytics.config['acquisition_interval']
        self.classes = Analytics.config['classes'] # person, bicycle, car, elephant, zebra
        self.target_classes = Analytics.config['target_classes'] # person, elephant

        ObjectDetector.init_backend()
        print('[Analytics] Loading acquisition detector model...')
        self.acq_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.ACQUISITION)
        # print('[Analytics] Loading tracking detector model...')
        # self.trk_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.TRACKING)
        self.tracker = KalmanTracker(self.size, capture_dt)
        
        # reset flags
        self.status = Analytics.Status.SEARCHING
        self.acquire = True
        self.detector = self.acq_detector
        self.detector_frame_skip = self.acq_detector_frame_skip
        self.acquisition_start_frame = 0
        self.track_id = None
        self.frame_count = 0
    
    def run(self, frame):
        detections = []
        if self.frame_count == 0:
            print('\n[Analytics] Acquiring new targets...')
            detections = self.detector.detect_sync(frame)
            self.tracker.init(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.preprocess(frame, self.tracker.tracks, track_id=self.track_id)
                self.detector.infer_async()
                self.tracker.track(frame)
                detections = self.detector.postprocess()
                # self.tracker.update(detections, [self.detector.cur_tile], self.detector.tile_overlap, acquire=self.acquire)
                self.tracker.update(detections, self.detector.tiles, self.detector.tile_overlap, self.detector.get_tiling_region(), acquire=self.acquire)
            else:
                self.tracker.track(frame)

        if self.enable_drawing:
            self._draw(frame, detections, debug=True)

        if self.acquire:
            if self.frame_count - self.acquisition_start_frame + 1 == self.acquisition_interval:
                self.track_id = self.tracker.get_nearest_track(self.target_classes)
                if self.track_id is not None:
                    self.acquire = False
                    self.detector = self.trk_detector
                    self.detector_frame_skip = self.trk_detector_frame_skip
                    self.status = Analytics.Status.TARGET_ACQUIRED
                    print('[Analytics] Following: %s' % self.tracker.tracks[self.track_id])
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
            self.status = Analytics.Status.TARGET_LOST
            print('[Analytics] Acquiring new targets...')
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

    def _draw(self, frame, detections, debug=False):
        for track_id, track in self.tracker.tracks.items():
            if self.status == Analytics.Status.TARGET_ACQUIRED and track_id == self.track_id:
                # track.draw(frame, follow=True, draw_feature_match=debug)
                track.draw(frame, follow=True, draw_feature_match=True)
            else:
                # track.draw(frame, draw_feature_match=debug)
                track.draw(frame, draw_feature_match=True)
        if debug:
            [det.draw(frame) for det in detections]
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_tile(frame)
        if self.acquire:
            cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        elif self.status == Analytics.Status.TARGET_ACQUIRED:
            cv2.putText(frame, 'Following %d' % self.track_id, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    