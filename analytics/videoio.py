from pathlib import Path
from collections import deque
import threading
import time
import json
import cv2

from .configs import decoder

class VideoIO:
    with open(Path(__file__).parent / 'configs' / 'config.json') as config_file:
        config = json.load(config_file, cls=decoder.decoder)['VideoIO']

    def __init__(self, size, input_path=None, output_path=None, delay=0):
        self.size = size
        self.input_path = input_path
        self.output_path = output_path
        self.delay = delay
        self.capture_size = VideoIO.config['capture_size']
        self.camera_fps = VideoIO.config['camera_fps']
        self.flip_method = VideoIO.config['flip_method']
        self.max_queue_size = VideoIO.config['max_queue_size']

        if self.input_path is None:
            # use camera when no input path is provided
            self.cap = cv2.VideoCapture(self._gst_cap_str(), cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.input_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.vid_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture_dt = 1 / self.fps
        if self.input_path is None:
            # delay for camera
            self.delay = self.capture_dt = max(self.delay, self.capture_dt)
            self.fps = 1 / self.capture_dt

        self.frame_queue = deque()
        self.cond = threading.Condition()
        self.exit_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read video stream")
        self.frame_queue.append(frame)
        print('[Video] Stream specs: %dx%d @ %d FPS' % (*self.vid_size, self.fps))
        
        if self.output_path is not None:
            assert Path(self.output_path).suffix == '.mp4', 'Only mp4 is supported'
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(self._gst_write_str(), 0, self.fps, self.size, True)

    def start_capture(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_cap_str(), cv2.CAP_GSTREAMER)
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

        if self.vid_size != self.size:
            frame = cv2.resize(frame, self.size)
        return frame

    def write(self, frame):
        assert hasattr(self, 'writer')
        self.writer.write(frame)

    def release(self):
        if self.input_path is not None:
            self.cap.release()
        if hasattr(self, 'writer'):
            self.writer.release()

    def _gst_cap_str(self):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx !"
            "videoconvert ! appsink"
            % (
                *self.capture_size,
                self.camera_fps,
                self.flip_method,
                *self.size
            )
    )

    def _gst_write_str(self):
        return 'appsrc ! autovideoconvert ! omxh265enc ! mp4mux ! filesink location = %s ' % self.output_path

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
                if self.delay - time_elapsed <= 0.01:
                    # ret, frame = self.cap.retrieve()
                    tic = time.time()
                    while len(self.frame_queue) >= self.max_queue_size and not self.exit_event.is_set():
                        self.cond.wait()
                    self.frame_queue.append(frame)
                    self.cond.notify()
