import cv2
import argparse
import time
import os
from pathlib import Path
import objectdetector
import bboxtracker
import threading

def gst_pipeline(
    capture_size=(1920, 1080),
    display_size=(1280, 720),
    frame_rate=30,
    flip_method=0,
):  
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert !"
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_size[0],
            capture_size[1],
            frame_rate,
            flip_method,
            display_size[0],
            display_size[1]
        )
    )


def draw(frame, detections, tracker):
    [track.draw(frame) for track in tracker.tracks.values()]
    [det.draw(frame) for det in detections]
    if tracker is not None:
        tracker.draw_bkg_feature_match(frame)


def capture_frames(cap, frame_queue, capture_rate=None):
    while not exit_event.is_set():
        ret, frame = cap.read()
        with cond:
            if ret == False:
                exit_event.set()
                cond.notify()
                break
            frame_queue.append(frame)
            cond.notify()
        if capture_rate is not None:
            time.sleep(1 / capture_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help='Path to optional input video file')
    parser.add_argument('-o', '--output', help='Path to optional output video file')
    parser.add_argument('-d', '--enable_detector', action='store_true', help='Turn on object detector')
    parser.add_argument('-g', '--gui', action='store_true', help='Turn on visiualization')
    parser.add_argument('-f', '--flip', type=int, default=0, choices=range(8), help=
        "0: none\n"          
        "1: counterclockwise\n"
        "2: rotate-180\n"    
        "3: clockwise\n" 
        "4: horizontal-flip\n"
        "5: upper-right-diagonal\n"
        "6: vertical-flip\n"
        "7: upper-left-diagonal\n"
        )
    args = vars(parser.parse_args())

    print('[INFO] Maximizing Nano Performance...')
    password = 'jdyw123'
    os.system('echo %s | sudo -S nvpmodel -m 0' % password)
    os.system('sudo jetson_clocks')

    print('[INFO] Starting video stream...') 
    detector = None
    tracker = None
    writer = None

    # paramters
    capture_size = (1920, 1080)
    camera_frame_rate = 30
    proc_size = (1280, 720)
    detector_frame_skip = 10
    # classes = set([1])
    classes = set([1, 2, 3, 22, 24]) # person, bicycle, car, elephant, zebra

    cap = cv2.VideoCapture(gst_pipeline(capture_size=capture_size, display_size=proc_size, frame_rate=camera_frame_rate, flip_method=args['flip']), cv2.CAP_GSTREAMER) if args['input'] is None else cv2.VideoCapture(args['input'])
    if cap.isOpened():
        cond = threading.Condition()
        exit_event = threading.Event()
        frame_queue = []
        elapsed_time = 0
        frame_count = 0
        capture_rate = camera_frame_rate if args['input'] is None else 40
        # account for display and output overhead
        if args['gui']:
            capture_rate = 1 / (1 / capture_rate + 0.02)
        if args['output'] is not None:
            capture_rate = 1 / (1 / capture_rate + 0.06)

        capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, capture_rate))
        if args['enable_detector']:
            print('[INFO] Loading detector model...')
            detector = objectdetector.ObjectDetector(proc_size, classes=classes)
            tracker = bboxtracker.BBoxTracker(proc_size, estimate_camera_motion=True)
            track_id = -1
        if args['output'] is not None:
            Path(args['output']).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, frame_rate, proc_size, True)
        print('[INFO] Starting capturing thread...')
        capture_thread.start()

        if args['gui']:
            cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
        try:
            while not args['gui'] or cv2.getWindowProperty("Video", 0) >= 0:
                tic = time.perf_counter()
                # grab frame when available
                with cond:
                    # print('frame queue size:', len(frame_queue))
                    while len(frame_queue) == 0 and not exit_event.is_set():
                        cond.wait()
                    if exit_event.is_set():
                        break
                    frame = frame_queue.pop(0)

                if frame_count == 0:
                    print('[INFO] Video dimension %dx%d' % (frame.shape[1], frame.shape[0]))
                if args['input'] is not None:
                    frame = cv2.resize(frame, proc_size)
                    # frame = cv2.medianBlur(frame, 3)
                
                if args['enable_detector']:
                    detections = []
                    if frame_count == 0:
                        detections = detector.detect_sync(frame)
                        tracker.init(frame, detections)
                    else:
                        if track_id not in tracker.tracks:
                            acquire = True
                        if frame_count % 100 == 0:
                            if len(tracker.tracks) > 0:
                                track_id = tracker.lock_on()
                                acquire = False

                        if frame_count % detector_frame_skip == 0:
                            detector.preprocess(frame, tracker.tracks, acquire=acquire)
                            detector.infer_async()
                        tracker.track(frame)
                        if frame_count % detector_frame_skip == 0:
                            detections = detector.postprocess()
                            tracker.update(detections, detector.roi, detector.tile_overlap, acquire=acquire)
                        # else:
                        #     tracks, H_camera = tracker.track(frame)
                            # frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
                            # trackers = []
                            # for det in detections:
                            #     tracker = cv2.TrackerKCF_create()
                            #     tracker.init(frame_small, det.bbox.scale(0.5, 0.5).cv_rect())
                            #     trackers.append(tracker)

                            # frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
                            # for i, tracker in enumerate(trackers):
                            #     success, bbox = tracker.update(frame_small)
                            #     if success:
                            #         (x, y, w, h) = [int(2 * v) for v in bbox]
                            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            #     else:
                            #         print('[INFO] Tracker failed')
                            #         del trackers[i]

                    if args['gui'] or args['output'] is not None:
                        draw(frame, detections, tracker)
                        if frame_count % detector_frame_skip == 0:
                            detector.roi.draw(frame)

                toc = time.perf_counter()
                fps = round(1 / (toc - tic))
                elapsed_time += toc - tic

                # tic = time.perf_counter()
                if args['gui']:
                    cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit_event.set()
                        break
                else:
                    print('[INFO] FPS: %d' % fps)
                # toc = time.perf_counter()
                # print('gui', toc - tic)

                # tic = time.perf_counter()
                if writer is not None:
                    writer.write(frame)
                # toc = time.perf_counter()
                # print('writer', toc - tic)
                frame_count += 1
        finally:
            # clean up resources
            if writer is not None:
                writer.release()
            if args['input'] is not None:
                # gst cleans up automatically
                cap.release()
            cv2.destroyAllWindows()

        avg_fps = round(frame_count / elapsed_time)
        print('[INFO] Average FPS: %d' % avg_fps)
    else:
        raise Exception("Unable to open video stream")
