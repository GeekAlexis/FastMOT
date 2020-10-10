from pathlib import Path
from enum import Enum
from collections import deque
import subprocess
import threading
import logging
import cv2


class Protocol(Enum):
    FILE = 0
    CSI = 1
    V4L2 = 2
    RTSP = 3


class VideoIO:
    """
    Class for video capturing from video files or cameras, and writing video files.
    Encoding and decoding are accelerated using the GStreamer backend.
    """
    
    def __init__(self, size, config, input_uri, output_uri=None, latency=1/30):
        self.size = size
        self.input_uri = input_uri
        self.output_uri = output_uri

        self.camera_size = config['camera_size']
        self.camera_fps = config['camera_fps']
        self.buffer_size = config['buffer_size']

        self.protocol = self._parse_uri(self.input_uri)
        self.cap = cv2.VideoCapture(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        self.frame_queue = deque([], maxlen=self.buffer_size)
        self.cond = threading.Condition()
        self.exit_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError('Unable to read video stream')
        self.frame_queue.append(frame)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = self.camera_fps # fallback
        self.capture_dt = 1 / self.fps
        logging.info('%dx%d stream @ %d FPS', *self.size, self.fps)

        output_fps = self.fps
        if self.protocol != Protocol.FILE:
            # limit capture latency at processing latency
            self.capture_dt = max(self.capture_dt, latency)
            output_fps = 1 / self.capture_dt
        if self.output_uri is not None:
            Path(self.output_uri).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(self._gst_write_pipeline(), 0, output_fps, self.size, True)

    def start_capture(self):
        """
        Start capturing from video file or device.
        """
        if not self.cap.isOpened():
            self.cap.open(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        if not self.capture_thread.is_alive():
            self.capture_thread.start()

    def stop_capture(self):
        """
        Stop capturing from video file or device.
        """
        with self.cond:
            self.exit_event.set()
            self.cond.notify()
        self.frame_queue.clear()
        self.capture_thread.join()

    def read(self):
        """
        Returns the next video frame.
        Returns None if there are no more frames.
        """
        with self.cond:
            while len(self.frame_queue) == 0 and not self.exit_event.is_set():
                self.cond.wait()
            if len(self.frame_queue) == 0 and self.exit_event.is_set():
                return None
            frame = self.frame_queue.popleft()
            self.cond.notify()
        return frame

    def write(self, frame):
        """
        Writes the next video frame.
        """
        assert hasattr(self, 'writer')
        self.writer.write(frame)

    def release(self):
        """
        Closes video file or capturing device.
        """
        self.stop_capture()
        if hasattr(self, 'writer'):
            self.writer.release()

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
        if 'nvvidconv' in gst_elements and self.protocol != Protocol.V4L2:
            # format conversion for hardware decoder
            cvt_pipeline = (
                'nvvidconv ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx !'
                'videoconvert ! appsink'
                % self.size
            )
        else:
            cvt_pipeline = (
                'videoscale ! '
                'video/x-raw, width=(int)%d, height=(int)%d !'
                'videoconvert ! appsink'
                % self.size
            )

        if self.protocol == Protocol.FILE:
            pipeline = 'filesrc location=%s ! decodebin ! ' % self.input_uri
        elif self.protocol == Protocol.CSI:
            if 'nvarguscamerasrc' in gst_elements:
                pipeline = (
                    'nvarguscamerasrc sensor_id=%s ! '
                    'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, '
                    'format=(string)NV12, framerate=(fraction)%d/1 ! '
                    % (
                        self.input_uri[6:],
                        *self.camera_size,
                        self.camera_fps
                    )
                )
            else:
                raise RuntimeError('GStreamer CSI plugin not found')
        elif self.protocol == Protocol.V4L2:
            if 'v4l2src' in gst_elements:
                pipeline = (
                    'v4l2src device=%s ! '
                    'video/x-raw, width=(int)%d, height=(int)%d, '
                    'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                    % (
                        self.input_uri,
                        *self.camera_size,
                        self.camera_fps
                    )
                )
            else:
                raise RuntimeError('GStreamer V4L2 plugin not found')
        elif self.protocol == Protocol.RTSP:
            pipeline = 'rtspsrc location=%s latency=0 ! decodebin ! ' % self.input_uri
        return pipeline + cvt_pipeline

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
            raise RuntimeError('GStreamer H.264 encoder not found')
        pipeline = (
            'appsrc ! autovideoconvert ! %s ! qtmux ! filesink location=%s '
            % (
                h264_encoder, 
                self.output_uri
            )
        )
        return pipeline

    def _capture_frames(self):
        while not self.exit_event.is_set():
            ret, frame = self.cap.read()
            with self.cond:
                if not ret:
                    self.exit_event.set()
                    self.cond.notify()
                    break
                # keep unprocessed frames in the buffer for video file
                if self.protocol == Protocol.FILE:
                    while len(self.frame_queue) == self.buffer_size and not self.exit_event.is_set():
                        self.cond.wait()
                self.frame_queue.append(frame)
                self.cond.notify()
