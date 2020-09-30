from pathlib import Path
from enum import Enum
from collections import deque
import subprocess
import threading
import logging
import time
import cv2


class Protocol(Enum):
    FILE = 0
    CSI = 1
    V4L2 = 2
    RTSP = 3


class VideoIO:
    def __init__(self, size, config, input_uri, output_uri=None, latency=0):
        self.size = size
        self.input_uri = input_uri
        self.output_uri = output_uri
        self.latency = latency

        self.camera_size = config['camera_size']
        self.camera_fps = config['camera_fps']
        self.max_queue_size = config['max_queue_size']

        self.protocol = self._parse_uri(self.input_uri)
        self.cap = cv2.VideoCapture(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        self.frame_queue = deque([], maxlen=self.max_queue_size)
        self.cond = threading.Condition()
        self.exit_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError('Unable to read video stream')
        self.frame_queue.append(frame)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.capture_dt = 1 / self.fps
        logging.info('%dx%d stream @ %d FPS', *self.size, self.fps)

        if self.protocol != Protocol.FILE:
            # delay for camera
            self.latency = self.capture_dt = max(self.latency, self.capture_dt)
            self.fps = 1 / self.capture_dt
        if self.output_uri is not None:
            Path(self.output_uri).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(self._gst_write_pipeline(), 0, self.fps, self.size, True)

    def start_capture(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        if not self.capture_thread.is_alive():
            self.capture_thread.start()

    def stop_capture(self):
        with self.cond:
            self.exit_event.set()
            self.cond.notify()
        self.frame_queue.clear()
        self.capture_thread.join()

    def read(self):
        with self.cond:
            # print('frame queue size:', len(self.frame_queue))
            while len(self.frame_queue) == 0 and not self.exit_event.is_set():
                self.cond.wait()
            if len(self.frame_queue) == 0 and self.exit_event.is_set():
                return None
            frame = self.frame_queue.popleft()
            self.cond.notify()
        return frame

    def write(self, frame):
        assert hasattr(self, 'writer')
        self.writer.write(frame)

    def release(self):
        if hasattr(self, 'writer'):
            self.writer.release()
        self.stop_capture()
        self.cap.release()
        # try:
        #     self.cap.release()
        # except Exception:
        #     pass

    def _parse_uri(self, uri):
        pos = uri.find('://')
        if '/dev/video' in uri:
            protocol = Protocol.V4L2
        elif uri[:pos] == 'csi':
            protocol = Protocol.CSI
        elif uri[:pos] == 'rtsp':
            protocol = Protocol.RTSP
        else:
            protocol = Protocol.FILE
        return protocol
        
    def _gst_cap_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        scaling_pad = 'nvvidconv' if 'nvvidconv' in gst_elements else 'videoscale'
        if self.protocol == Protocol.FILE:
            gst_pipeline = (
                'filesrc location=%s ! '
                'decodebin ! '
                '%s ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                'videoconvert ! appsink'
                % (
                    self.input_uri,
                    scaling_pad,
                    *self.size
                )
            )
        elif self.protocol == Protocol.CSI:
            if 'nvarguscamerasrc' in gst_elements:
                gst_pipeline = (
                    'nvarguscamerasrc sensor_id=%s ! '
                    'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, '
                    'format=(string)NV12, framerate=(fraction)%d/1 ! '
                    'nvvidconv flip-method=2 ! '
                    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                    'videoconvert ! appsink'
                    % (
                        self.input_uri[6:],
                        *self.camera_size,
                        self.camera_fps,
                        *self.size
                    )
                )
            else:
                raise RuntimeError('Gstreamer CSI plugin not found')
        elif self.protocol == Protocol.V4L2:
            if 'v4l2src' in gst_elements:
                gst_pipeline = (
                    'v4l2src device=%s ! '
                    'video/x-raw, width=(int)%d, height=(int)%d ! '
                    'videorate ! video/x-raw, framerate=(fraction)%d/1 ! '
                    'videoconvert ! appsink'
                    % (
                        self.input_uri,
                        *self.size,
                        self.camera_fps
                    )
                )
                # gst_pipeline = (
                #     'v4l2src device=/dev/video%d ! '
                #     'video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! '
                #     'videoscale ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx !'
                #     'videoconvert ! appsink'
                #     % (
                #         self.camera_id,
                #         *self.camera_size,
                #         self.camera_fps,
                #         *self.size
                #     )
                # )
            else:
                raise RuntimeError('Gstreamer V4L2 plugin not found')
        elif self.protocol == Protocol.RTSP:
            gst_pipeline = (
                'rtspsrc location=%s ! '
                'decodebin ! '
                '%s ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                'videoconvert ! appsink'
                % (
                    self.input_uri,
                    scaling_pad,
                    *self.size
                )
            )
        return gst_pipeline
    # 'rtspsrc location=rtsp://<user>:<pass>@<ip>:<port> ! 'decodebin' ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink'
    # 'rtph264depay ! h264parse ! omxh264dec'

    def _gst_write_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        # use hardware encoder if found
        if 'omxh264enc' in gst_elements:
            h264_encoder = 'omxh264enc'
        elif 'avenc_h264_omx' in gst_elements:
            h264_encoder = 'avenc_h264_omx'
        elif 'x264enc' in gst_elements:
            h264_encoder = 'x264enc'
        else:
            raise RuntimeError('Gstreamer H.264 encoder not found')
        gst_pipeline = 'appsrc ! autovideoconvert ! %s ! qtmux ! filesink location=%s ' % (h264_encoder, self.output_uri)
        return gst_pipeline

    def _capture_frames(self):
        tic = time.time()
        while not self.exit_event.is_set():
            ret, frame = self.cap.read()
            # ret = self.cap.grab()
            with self.cond:
                if not ret:
                    self.exit_event.set()
                    self.cond.notify()
                    break
                time_elapsed = time.time() - tic
                if self.latency - time_elapsed <= 0.01:
                    # ret, frame = self.cap.retrieve()
                    tic = time.time()
                    while len(self.frame_queue) == self.max_queue_size and not self.exit_event.is_set():
                        self.cond.wait()
                    self.frame_queue.append(frame)
                    self.cond.notify()
