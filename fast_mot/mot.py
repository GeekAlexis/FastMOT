from enum import Enum
from pathlib import Path
import logging
import json
import cv2
import time

from .detector import ObjectDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import ConfigDecoder, draw_trk, draw_det


class Mot:
    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['Mot']

    def __init__(self, size, capture_dt, enable_drawing=False, verbose=False):
        self.size = size
        self.enable_drawing = enable_drawing
        self.verbose = verbose
        self.detector_frame_skip = Mot.config['detector_frame_skip']
        self.classes = Mot.config['classes']

        logging.info('Loading detector model...')
        self.detector = ObjectDetector(self.size, self.classes)
        logging.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor()
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric)
        
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
        if self.frame_count == 0:
            detections = self.detector(frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                tic2 = time.perf_counter()
                self.detector.detect_async(frame)
                elapsed = time.perf_counter() - tic2
                self.det_pre_time += elapsed
                logging.debug('detect pre %f', elapsed)
                tic2 = time.perf_counter()
                self.tracker.compute_flow(frame)
                detections = self.detector.postprocess()
                elapsed = time.perf_counter() - tic2
                self.det_time += elapsed
                logging.debug('detect / flow %f', elapsed)
                tic2 = time.perf_counter()
                self.extractor.extract_async(frame, detections)
                self.tracker.step_kalman_filter()
                embeddings = self.extractor.postprocess()
                elapsed = time.perf_counter() - tic2
                self.embedding_time += elapsed
                logging.debug('embedding / kf %f', elapsed)
                tic2 = time.perf_counter()
                self.tracker.update(detections, embeddings)
                elapsed = time.perf_counter() - tic2
                self.match_time += elapsed
                logging.debug('match %f', elapsed)
                logging.debug('UPDATE %f', time.perf_counter() - tic)
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)
                elapsed = time.perf_counter() - tic
                self.track_time += elapsed
                logging.debug('TRACK %f', elapsed)

        if self.enable_drawing:
            self._draw(frame, detections, debug=self.verbose)

        self.frame_count += 1

    def initiate(self):
        self.frame_count = 0

    def _draw(self, frame, detections, debug=False):
        count = 0
        for track in self.tracker.tracks.values():
            if track.confirmed and track.active:
                draw_trk(frame, track, draw_feature_match=debug)
                count += 1
        if debug:
            [draw_det(frame, det) for det in detections]
            # self.tracker.flow.draw_bkg_feature_match(frame)
            if self.frame_count % self.detector_frame_skip == 0:
                self.detector.draw_tile(frame)
        cv2.putText(frame, f'visible: {count}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
