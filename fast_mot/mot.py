from enum import Enum
from pathlib import Path
import json
import cv2
import time

from .detector import ObjectDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import ConfigDecoder


class Mot:
    
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['Mot']

    def __init__(self, size, capture_dt, enable_drawing=False):
        self.size = size
        self.enable_drawing = enable_drawing
        self.detector_frame_skip = Mot.config['detector_frame_skip']
        self.classes = Mot.config['classes']

        print('[INFO] Loading detector model...')
        self.detector = ObjectDetector(self.size, self.classes)
        print('[INFO] Loading feature extractor model...')
        self.extractor = FeatureExtractor()
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric, self.detector.tiling_region)
        
        # reset flags
        self.frame_count = 0
        self.detector_frame_count = 0
        self.det_pre_time = 0
        self.det_time = 0
        self.embedding_time = 0
        self.match_time = 0
        self.track_time = 0
    
    def run(self, frame):
        detections = []
        orig_dets = []
        if self.frame_count == 0:
            detections, orig_dets = self.detector(frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                tic2 = time.perf_counter()
                self.detector.detect_async(frame)
                elapsed = time.perf_counter() - tic2
                self.det_pre_time += elapsed
                print('detector pre', elapsed)
                tic2 = time.perf_counter()
                self.tracker.track(frame)
                detections, orig_dets = self.detector.postprocess()
                elapsed = time.perf_counter() - tic2
                self.det_time += elapsed
                print('det / track + post', elapsed)
                tic2 = time.perf_counter()
                embeddings = self.extractor(frame, detections)
                elapsed = time.perf_counter() - tic2
                self.embedding_time += elapsed
                print('embedding', elapsed)
                tic2 = time.perf_counter()
                self.tracker.update(detections, embeddings)
                elapsed = time.perf_counter() - tic2
                self.match_time += elapsed
                print('match', elapsed)
                print('UPDATE', time.perf_counter() - tic)
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)
                elapsed = time.perf_counter() - tic
                self.track_time += elapsed
                print('TRACK', elapsed)

        if self.enable_drawing:
            self._draw(frame, orig_dets, debug=True)

        self.frame_count += 1

    def initiate(self):
        self.frame_count = 0

    def _draw(self, frame, detections, debug=False):
        for track in self.tracker.tracks.values():
            if track.confirmed and track.age <= 2:
                track.draw(frame, draw_feature_match=debug)
        if debug:
            [det.draw(frame) for det in detections]
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_tile(frame)
        # cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)