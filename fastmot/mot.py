from enum import Enum
import logging
import cv2

from .detector import SSDDetector, YoloDetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    cap_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self, size, cap_dt, config, draw=False, verbose=False):
        self.size = size
        self.draw = draw
        self.verbose = verbose
        self.detector_type = DetectorType[config['detector_type']]
        self.detector_frame_skip = config['detector_frame_skip']

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, config['ssd_detector'])
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YoloDetector(self.size, config['yolo_detector'])
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, self.detector_frame_skip,
                                           config['public_detector'])

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(config['feature_extractor'])
        self.tracker = MultiTracker(self.size, cap_dt, self.extractor.metric,
                                    config['multi_tracker'])

        # reset counter
        self.frame_count = 0

    @property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def initiate(self):
        """
        Resets multiple object tracker.
        """
        self.frame_count = 0

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            detections = self.detector(frame)
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                with Profiler('preproc'):
                    self.detector.detect_async(frame)

                with Profiler('detect'):
                    with Profiler('track'):
                        self.tracker.compute_flow(frame)
                    detections = self.detector.postprocess()

                with Profiler('extract'):
                    self.extractor.extract_async(frame, detections)
                    with Profiler('track', aggregate=True):
                        self.tracker.apply_kalman()
                    embeddings = self.extractor.postprocess()

                with Profiler('assoc'):
                    self.tracker.update(self.frame_count, detections, embeddings)
            else:
                with Profiler('track'):
                    self.tracker.track(frame)

        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    @staticmethod
    def print_timing_info():
        LOGGER.debug('===============Timing Stats===============')
        LOGGER.debug('track time:            %.3f ms', Profiler.get_avg_millis('track'))
        LOGGER.debug('feature extract time:  %.3f ms', Profiler.get_avg_millis('extract'))
        LOGGER.debug('preprocess time:       %.3f ms', Profiler.get_avg_millis('preproc'))
        LOGGER.debug('detect time:           %.3f ms', Profiler.get_avg_millis('detect'))
        LOGGER.debug('association time:      %.3f ms', Profiler.get_avg_millis('assoc'))

    def _draw(self, frame, detections):
        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'visible: {len(self.visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
