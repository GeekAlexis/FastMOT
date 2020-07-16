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
        print('[INFO] Loading encoder model...')
        self.extractor = FeatureExtractor()
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric, self.detector.tiling_region)
        
        # reset flags
        self.frame_count = 0
    
    def run(self, frame):
        detections = []
        if self.frame_count == 0:
            detections = self.detector.detect(frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                tic2 = time.perf_counter()
                self.detector.detect_async(frame)
                print('detector pre', time.perf_counter() - tic2)
                tic2 = time.perf_counter()
                self.tracker.track(frame)
                detections = self.detector.postprocess()
                print('det / track + post', time.perf_counter() - tic2)
                tic2 = time.perf_counter()
                embeddings = self.extractor(frame, detections)
                print('embedding', time.perf_counter() - tic2)
                tic2 = time.perf_counter()
                self.tracker.update(detections, embeddings, frame)
                print('match', time.perf_counter() - tic2)
                print('UPDATE', time.perf_counter() - tic)
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)
                print('TRACK', time.perf_counter() - tic)

        if self.enable_drawing:
            self._draw(frame, detections, debug=True)

        self.frame_count += 1

    def initiate(self):
        self.frame_count = 0

    def _draw(self, frame, detections, debug=False):
        for track in self.tracker.tracks.values():
            if track.confirmed and track.age == 0:
                track.draw(frame, draw_feature_match=debug)
        if debug:
            [det.draw(frame) for det in detections]
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_tile(frame)
        # cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)