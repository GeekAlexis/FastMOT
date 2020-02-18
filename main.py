import argparse
import time
import os
from pathlib import Path
import threading
import cv2
import objectdetector
import bboxtracker


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


def draw(frame, detections, tracks, acquire, track_id):
    for _track_id, track in tracks.items():
        if not acquire and _track_id == track_id:
            track.draw(frame, follow=True, draw_feature_match=True)
        else:
            track.draw(frame, draw_feature_match=True)
    [det.draw(frame) for det in detections]
    if acquire:
        cv2.putText(frame, 'Acquiring', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
    elif track_id in tracks:
        cv2.putText(frame, 'Tracking %d' % track_id, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
        

def capture_frames(cap, frame_queue, cond, exit_event, capture_delay=None):
    while not exit_event.is_set():
        ret, frame = cap.read()
        with cond:
            if ret == False:
                exit_event.set()
                cond.notify()
                break
            frame_queue.append(frame)
            cond.notify()
        if capture_delay is not None:
            time.sleep(capture_delay)


def main():
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
    acq_detector_frame_skip = 4
    trk_detector_frame_skip = 6
    # classes = set([1])
    classes = set([1, 2, 3, 22, 24]) # person, bicycle, car, elephant, zebra

    cap = cv2.VideoCapture(gst_pipeline(capture_size, proc_size, camera_frame_rate, args['flip']), cv2.CAP_GSTREAMER) if args['input'] is None else cv2.VideoCapture(args['input'])
    if cap.isOpened():
        cond = threading.Condition()
        exit_event = threading.Event()
        frame_queue = []
        elapsed_time = 0
        frame_count = 0

        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        capture_dt = 1 / vid_fps
        # account for display and output overhead so that capture queue won't overflow
        capture_delay = capture_dt if args['input'] is None else 1 / 40 # video has max performance at 40 FPS
        if args['gui']:
            capture_delay += 0.025
        if args['output'] is not None:
            capture_delay += 0.06
        if args['input'] is None:
            # camera only grabs the most recent frame
            capture_dt = capture_delay

        capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, cond, exit_event, capture_delay))
        if args['enable_detector']:
            objectdetector.ObjectDetector.init_backend()
            print('[INFO] Loading tracking detector model...')
            acq_detector = objectdetector.ObjectDetector(proc_size, classes, objectdetector.DetectorType.ACQUISITION)
            print('[INFO] Loading acquisition detector model...')
            trk_detector = objectdetector.ObjectDetector(proc_size, classes, objectdetector.DetectorType.TRACKING)
            tracker = bboxtracker.BBoxTracker(proc_size, capture_dt)
            acquire = True
            detector = acq_detector
            detector_frame_skip = acq_detector_frame_skip
            acquisition_start_frame = 0
            track_id = -1
        if args['output'] is not None:
            Path(args['output']).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, vid_fps, proc_size, True)
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
                    print('[INFO] Video specs: %dx%d @ %d FPS' % (frame.shape[1], frame.shape[0], vid_fps))
                if frame.shape[:2][::-1] != proc_size:
                    frame = cv2.resize(frame, proc_size)
                    # frame = cv2.medianBlur(frame, 3)
                
                if args['enable_detector']:
                    detections = []
                    if frame_count == 0:
                        detections = detector.detect_sync(frame)
                        tracker.init(frame, detections)
                        print('[INFO] Acquiring new targets...')
                    else:
                        if acquire:
                            if frame_count - acquisition_start_frame == 100:
                                if len(tracker.tracks) > 0:
                                    track_id = tracker.get_nearest_track()
                                    acquire = False
                                    detector = trk_detector
                                    detector_frame_skip = trk_detector_frame_skip
                                    print('[INFO] Following: %s' % tracker.tracks[track_id])
                                else:
                                    acquisition_start_frame = frame_count
                        elif track_id not in tracker.tracks:
                            acquire = True
                            detector = acq_detector
                            detector_frame_skip = acq_detector_frame_skip
                            acquisition_start_frame = frame_count
                            print('[INFO] Acquiring new targets...')

                        if frame_count % detector_frame_skip == 0:
                            # tic = time.perf_counter()
                            detector.preprocess(frame, tracker.tracks, track_id=track_id)
                            detector.infer_async()
                            tracker.track(frame)
                            detections = detector.postprocess()
                            # print('[INFO] Detector Latency: %.3f' % (time.perf_counter() - tic))
                            tracker.update(detections, detector.cur_tile, detector.tile_overlap, acquire=acquire)
                        else:
                            tracker.track(frame)

                    if args['gui'] or args['output'] is not None:
                        # draw tracks and detections
                        draw(frame, detections, tracker.tracks, acquire, track_id)
                        tracker.flow_tracker.draw_bkg_feature_match(frame)
                        if frame_count % detector_frame_skip == 0:
                            detector.cur_tile.draw(frame)

                toc = time.perf_counter()
                # fps = round(1 / (toc - tic))
                elapsed_time += toc - tic

                if args['gui']:
                    # cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        exit_event.set()
                        break
                # else:
                #     print('[INFO] FPS: %d' % fps)

                if writer is not None:
                    writer.write(frame)
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


if __name__ == '__main__':
    main()
