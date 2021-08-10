from types import SimpleNamespace
from enum import Enum
import logging
import cv2

from .detector import SSDDetector, YOLODetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_klt_bboxes, draw_background_flow


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    def __init__(self, size,
                 detector_type='YOLO',
                 detector_frame_skip=5,
                 ssd_detector_cfg=None,
                 yolo_detector_cfg=None,
                 public_detector_cfg=None,
                 feature_extractor_cfg=None,
                 tracker_cfg=None,
                 draw=False,
                 verbose=False):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'PUBLIC'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfg : SimpleNamespace, optional
            Feature extractor configuration.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        draw : bool, optional
            Enable drawing tracks.
        verbose : bool, optional
            Enable verbose drawing.
        """
        self.size = size
        self.detector_type = DetectorType[detector_type.upper()]
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.draw = draw
        self.verbose = verbose

        if ssd_detector_cfg is None:
            ssd_detector_cfg = SimpleNamespace()
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if public_detector_cfg is None:
            public_detector_cfg = SimpleNamespace()
        if feature_extractor_cfg is None:
            feature_extractor_cfg = SimpleNamespace()
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, **vars(ssd_detector_cfg))
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YOLODetector(self.size, **vars(yolo_detector_cfg))
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, self.detector_frame_skip,
                                           **vars(public_detector_cfg))

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(**vars(feature_extractor_cfg))
        self.tracker = MultiTracker(self.size, self.extractor.metric, **vars(tracker_cfg))
        self.frame_count = 0

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker
        """
        # return (track for track in self.tracker.tracks.values()
        #         if track.confirmed and track.active)
        return (track for track in self.tracker.tracks.values()
                if track.confirmed)

    def reset(self, cap_dt):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        self.tracker.reset(cap_dt)

    def step(self, frame):
        """Runs multiple object tracker on the next frame.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            detections = self.detector(frame)
            self.tracker.init(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                with Profiler('preproc'):
                    self.detector.detect_async(frame)

                with Profiler('detect'):
                    with Profiler('track'):
                        self.tracker.compute_flow(frame)
                    detections = self.detector.postprocess()

                with Profiler('extract'):
                    self.extractor.extract_async(frame, detections.tlbr)
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
        LOGGER.debug('=================Timing Stats=================')
        LOGGER.debug(f"{'track time:':<37}{Profiler.get_avg_millis('track'):>6.3f} ms")
        LOGGER.debug(f"{'preprocess time:':<37}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        LOGGER.debug(f"{'detect/flow time:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        LOGGER.debug(f"{'feature extract/kalman filter time:':<37}"
                     f"{Profiler.get_avg_millis('extract'):>6.3f} ms")
        LOGGER.debug(f"{'association time:':<37}{Profiler.get_avg_millis('assoc'):>6.3f} ms")
        LOGGER.debug(f"{'match time:':<37}{Profiler.get_avg_millis('match'):>6.3f} ms")
        LOGGER.debug(f"{'match time1:':<37}{Profiler.get_avg_millis('match1'):>6.3f} ms")
        LOGGER.debug(f"{'match time2:':<37}{Profiler.get_avg_millis('match2'):>6.3f} ms")
        LOGGER.debug(f"{'iou cost time:':<37}{Profiler.get_avg_millis('iou_cost'):>6.3f} ms")
        LOGGER.debug(f"{'update time:':<37}{Profiler.get_avg_millis('update'):>6.3f} ms")

    def _draw(self, frame, detections):
        visible_tracks = list(self.visible_tracks())
        draw_tracks(frame, visible_tracks, show_flow=self.verbose, show_cov=True)
        if self.verbose:
            draw_detections(frame, detections)
            draw_klt_bboxes(frame, self.tracker.klt_bboxes.values())
            draw_background_flow(frame, self.tracker.flow.prev_bg_keypoints,
                                 self.tracker.flow.bg_keypoints)
        cv2.putText(frame, f'visible: {len(visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
