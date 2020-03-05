import argparse
import time
import os
import socket
import errno

import cv2
import videoio
import objectdetector
import kalmantracker


"""
config paramters
"""
MSG_LENGTH = 16
PASSWORD = 'jdyw123'
PROC_SIZE = (1280, 720)
ACQ_DETECTOR_FRAME_SKIP = 3
TRK_DETECTOR_FRAME_SKIP = 5
ACQUISITION_INTERVAL = 100
# CLASSES = set([1]) # person only
CLASSES = set([1, 2, 3, 22, 24]) # person, bicycle, car, elephant, zebra


class Msg:
    """
    enumeration type and function for socket messages
    """
    TARGET_NOT_FOUND, TARGET_ACQUIRED, TARGET_LOST, START, STOP, TERMINATE = ((i).to_bytes(MSG_LENGTH, byteorder='big') for i in range(6))


def convert_bbox_to_bytes(bbox):
    length = MSG_LENGTH // 4
    return b''.join(int(coord).to_bytes(length, byteorder='big') for coord in bbox.tf_rect())


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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help='Path to optional input video file')
    parser.add_argument('-o', '--output', help='Path to optional output video file')
    parser.add_argument('-a', '--analytics', action='store_true', help='Turn on video analytics')
    parser.add_argument('-s', '--socket', action='store_true', help='Turn on socket communication')
    parser.add_argument('--addr', default='/tmp/guardian_socket', help='Socket address')
    parser.add_argument('-g', '--gui', action='store_true', help='Turn on visiualization')
    # parser.add_argument('-f', '--flip', type=int, default=0, choices=range(8), help=
    #     "0: none\n"          
    #     "1: counterclockwise\n"
    #     "2: rotate-180\n"    
    #     "3: clockwise\n" 
    #     "4: horizontal-flip\n"
    #     "5: upper-right-diagonal\n"
    #     "6: vertical-flip\n"
    #     "7: upper-left-diagonal\n"
    #     )
    args = vars(parser.parse_args())

    print('[INFO] Maximizing Nano Performance...')
    os.system('echo %s | sudo -S nvpmodel -m 0' % PASSWORD)
    os.system('sudo jetson_clocks')

    delay = 0
    # Hack: delay camera frame grabbing to reduce lag
    if args['input'] is None:
        if args['analytics']:
            delay = 1 / 30 # main loop time
        if args['gui']:
            delay += 0.025 if args['analytics'] else 0.055 # gui time
    stream = videoio.VideoIO(PROC_SIZE, args['input'], args['output'], delay)
    
    detector = None
    tracker = None
    sock = None
    enable_analytics = False
    frame_count = 0
    elapsed_time = 0
    
    gui_time = 0
    draw_time = 0
    save_time = 0

    if args['analytics']:
        objectdetector.ObjectDetector.init_backend()
        print('[INFO] Loading acquisition detector model...')
        acq_detector = objectdetector.ObjectDetector(PROC_SIZE, CLASSES, objectdetector.DetectorType.ACQUISITION)
        print('[INFO] Loading tracking detector model...')
        trk_detector = objectdetector.ObjectDetector(PROC_SIZE, CLASSES, objectdetector.DetectorType.TRACKING)
        tracker = kalmantracker.KalmanTracker(PROC_SIZE, stream.capture_dt)
        # reset flags
        enable_analytics = True
        acquire = True
        detector = acq_detector
        detector_frame_skip = ACQ_DETECTOR_FRAME_SKIP
        acquisition_start_frame = 0
        track_id = -1
    if args['socket']:
        assert args['analytics'], 'Analytics must be turned on for socket'
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args['addr'])
        sock.setblocking(False)
        enable_analytics = False
    if args['gui']:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    print('[INFO] Starting video capture...')
    stream.start_capture()

    try:
        while not args['gui'] or cv2.getWindowProperty("Video", 0) >= 0:
            tic = time.perf_counter()
            frame = stream.read()
            if frame is None:
                break
            # frame = cv2.medianBlur(frame, 3)

            if args['socket']:
                try:
                    msg = sock.recv(MSG_LENGTH)
                except OSError as err:
                    if err.args[0] != errno.EAGAIN and err.args[0] != errno.EWOULDBLOCK:
                        raise
                else:
                    # print('client', msg)
                    if msg == Msg.START:
                        print('client: start')
                        if not enable_analytics:
                            # reset flags
                            frame_count = 0
                            elapsed_time = 0
                            enable_analytics = True
                            acquire = True
                            detector = acq_detector
                            detector_frame_skip = ACQ_DETECTOR_FRAME_SKIP
                            acquisition_start_frame = 0
                            track_id = -1
                    elif msg == Msg.STOP:
                        print('client: stop')
                        if enable_analytics:
                            enable_analytics = False
                            avg_fps = round(frame_count / elapsed_time)
                            print('[INFO] Average FPS: %d' % avg_fps)
                    elif msg == Msg.TERMINATE:
                        print('client: terminate')
                        stream.stop_capture()
                        break

            if enable_analytics:
                detections = []
                if frame_count == 0:
                    detections = detector.detect_sync(frame)
                    tracker.init(frame, detections)
                    print('\n[INFO] Acquiring new targets...')
                else:
                    if acquire:
                        if frame_count - acquisition_start_frame == ACQUISITION_INTERVAL:
                            if len(tracker.tracks) > 0:
                                track_id = tracker.get_nearest_track()
                                acquire = False
                                detector = trk_detector
                                detector_frame_skip = TRK_DETECTOR_FRAME_SKIP
                                print('[INFO] Following: %s' % tracker.tracks[track_id])
                                if args['socket']:
                                    sock.sendall(Msg.TARGET_ACQUIRED)
                            else:
                                acquisition_start_frame = frame_count
                                if args['socket']:
                                    sock.sendall(Msg.TARGET_NOT_FOUND)
                    elif track_id in tracker.tracks:
                        if args['socket']:
                            sock.sendall(convert_bbox_to_bytes(tracker.tracks[track_id].bbox))
                    else:
                        acquire = True
                        detector = acq_detector
                        detector_frame_skip = ACQ_DETECTOR_FRAME_SKIP
                        acquisition_start_frame = frame_count
                        print('[INFO] Acquiring new targets...')
                        if args['socket']:
                            sock.sendall(Msg.TARGET_LOST)

                    if frame_count % detector_frame_skip == 0:
                        detector.preprocess(frame, tracker.tracks, track_id=track_id)
                        detector.infer_async()
                        tracker.track(frame)
                        detections = detector.postprocess()
                        tracker.update(detections, detector.cur_tile, detector.tile_overlap, acquire=acquire)
                    else:
                        tracker.track(frame)

                if args['gui'] or args['output']:
                    tic2 = time.perf_counter()
                    draw(frame, detections, tracker.tracks, acquire, track_id)
                    # tracker.flow.draw_bkg_feature_match(frame)
                    if frame_count % detector_frame_skip == 0:
                        detector.draw_cur_tile(frame)
                    draw_time += time.perf_counter() - tic2

            if args['gui']:
                tic2 = time.perf_counter()
                # cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stream.stop_capture()
                    break
                gui_time += time.perf_counter() - tic2
            if args['output'] is not None:
                stream.write(frame)
            
            toc = time.perf_counter()
            elapsed_time += toc - tic
            # print('fps', 1 / (toc - tic))
            frame_count += 1
    finally:
        # clean up resources
        stream.release()
        if sock is not None:
            sock.close()
        cv2.destroyAllWindows()
    
    if not args['socket']:
        avg_fps = round(frame_count / elapsed_time)
        print('[INFO] Average FPS: %d' % avg_fps)
        if args['gui'] and args['analytics']:
            avg_time = draw_time / frame_count
            print('[INFO] Average drawing time: %f' % avg_time)
        if args['gui']:
            avg_time = gui_time / frame_count
            print('[INFO] Average GUI time: %f' % avg_time)
        # if args['output']:
        #     avg_time = save_time / frame_count
        #     print('[INFO] Average saving time: %f' % avg_time)


if __name__ == '__main__':
    main()
