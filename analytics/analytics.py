from enum import Enum
from pathlib import Path
import json
import cv2
import time

from .detector import ObjectDetector
from .encoder import ImageEncoder
from .tracker import MultiTracker
from .utils import ConfigDecoder


class Analytics:
    class Status(Enum):
        SEARCHING, TARGET_NOT_FOUND, TARGET_ACQUIRED, TARGET_LOST = (i for i in range(4))
    
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['Analytics']

    def __init__(self, size, capture_dt, enable_drawing=False):
        self.size = size
        self.enable_drawing = enable_drawing
        self.acq_detector_frame_skip = Analytics.config['acq_detector_frame_skip']
        self.trk_detector_frame_skip = Analytics.config['trk_detector_frame_skip']
        self.acquisition_interval = Analytics.config['acquisition_interval']
        self.classes = Analytics.config['classes']
        self.target_classes = Analytics.config['target_classes']

        print('[Analytics] Loading acquisition detector model...')
        self.acq_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.ACQUISITION)
        # print('[Analytics] Loading tracking detector model...')
        # self.trk_detector = ObjectDetector(self.size, self.classes, ObjectDetector.Type.TRACKING)
        print('[Analytics] Loading encoder model...')
        self.encoder = ImageEncoder()
        self.tracker = MultiTracker(self.size, capture_dt)
        
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
            detections = self.detector.detect(frame)
            embeddings = self.encoder.encode(frame, detections)
            self.tracker.initiate(frame, detections, embeddings)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                # tic = time.perf_counter()
                self.detector.detect_async(frame, roi=self.get_target_bbox())
                # print('det_pre', time.perf_counter() - tic)
                # self.tracker.track(frame, use_flow=True)
                tic = time.perf_counter()
                self.tracker.step_flow(frame)
                print('flow', time.perf_counter() - tic)
                # tic = time.perf_counter()
                detections = self.detector.postprocess()
                # print('det post', time.perf_counter() - tic)
                # tic = time.perf_counter()
                self.encoder.encode_async(frame, detections)
                # print('encode_pre', time.perf_counter() - tic)
                tic = time.perf_counter()
                self.tracker.step_kalman_filter(use_flow=True)
                print('kalman filter', time.perf_counter() - tic)
                # tic = time.perf_counter()
                embeddings = self.encoder.postprocess()
                # print('encode_post', time.perf_counter() - tic)
                # tic = time.perf_counter()
                self.tracker.update(detections, embeddings, self.detector.cur_tile, self.detector.tile_overlap, self.acquire)
                # print('update', time.perf_counter() - tic)
            else:
                self.tracker.track(frame, use_flow=True)

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
        if self.track_id is None:
            return
        return self.tracker.tracks[self.track_id].bbox

    def _draw(self, frame, detections, debug=False):
        for track_id, track in self.tracker.tracks.items():
            if self.status == Analytics.Status.TARGET_ACQUIRED and track_id == self.track_id:
                track.draw(frame, follow=True, draw_feature_match=debug)
                # track.draw(frame, follow=True, draw_feature_match=True)
            else:
                track.draw(frame, draw_feature_match=debug)
                # track.draw(frame, draw_feature_match=True)
        if debug:
            [det.draw(frame) for det in detections]
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_tile(frame)
        if self.acquire:
            cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        elif self.status == Analytics.Status.TARGET_ACQUIRED:
            cv2.putText(frame, 'Following %d' % self.track_id, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    