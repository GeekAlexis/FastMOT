import cv2
import argparse
import time
import os
from pathlib import Path
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


def draw(frame, detections, tracks, tracker):
    # [det.draw(frame) for det in detections]
    [track.draw(frame) for track in tracks.values()]
    if tracker is not None:
        tracker.draw_bkg_feature_match(frame)


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
    frame_rate = 30
    proc_size = (1280, 720)
    detector_frame_skip = 60
    # classes = set([1])
    classes = set([1, 2, 3, 22, 24]) # person, bicycle, car, elephant, zebra

    cap = cv2.VideoCapture(gst_pipeline(capture_size=capture_size, display_size=proc_size, frame_rate=frame_rate, flip_method=args['flip']), cv2.CAP_GSTREAMER) if args['input'] is None else cv2.VideoCapture(args['input'])
    if cap.isOpened():
        elapsed_time = 0
        frame_count = 0
        
        if args['gui']:
            cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
        try:
            while not args['gui'] or cv2.getWindowProperty("Video", 0) >= 0:
                tic = time.time()
                ret, frame = cap.read() # TODO: capture thread with queue
                if ret == False:
                    break
                if frame_count == 0:
                    print('[INFO] Video dimension %dx%d' % (frame.shape[1], frame.shape[0]))
                    if args['enable_detector']:
                        print('[INFO] Loading detector model...')
                        detector = objectdetector.ObjectDetector(proc_size, classes=classes)
                        tracker = bboxtracker.BBoxTracker(proc_size, estimate_camera_motion=True)
                    if args['output'] is not None:
                        Path(args['output']).parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args["output"], fourcc, 30, proc_size, True)

                if args['input'] is not None:
                    frame = cv2.resize(frame, proc_size)
                    # frame = cv2.medianBlur(frame, 3)
                
                detections = []
                tracks = {}
                if args['enable_detector']:
                    if frame_count % detector_frame_skip == 0:
                        detections = detector.detect(frame)
                        if frame_count == 0:
                            tracker.init(frame, detections)
                        else:
                            tracks = tracker.update(frame, detections, detector.get_cur_tile(scale_for_overlap=True))
                        detector.draw_cur_tile(frame)
                        # frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
                        # trackers = []
                        # for det in detections:
                        #     tracker = cv2.TrackerKCF_create()
                        #     tracker.init(frame_small, det.bbox.scale(0.5, 0.5).cv_rect())
                        #     trackers.append(tracker)
                    else:
                        tracks, H_camera = tracker.track(frame)
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
                    draw(frame, detections, tracks, tracker)

                toc = time.time()
                fps = round(1 / (toc - tic))
                elapsed_time += toc - tic
                if args['gui']:
                    cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    print('[INFO] FPS: %d' % fps)

                if writer is not None:
                    writer.write(frame)
                frame_count += 1
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            cv2.destroyAllWindows()

        avg_fps = round(frame_count / elapsed_time)
        print('[INFO] Average FPS: %d' % avg_fps)
    else:
        raise Exception("Unable to open video stream")