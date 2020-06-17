#!/usr/bin/env python3
import argparse
import time
import os
import socket
import struct
import errno
import cv2

from analytics import VideoIO
from analytics import Analytics


"""
constants
"""
MSG_LENGTH = 2
PROC_SIZE = (1280, 720)


class MsgType:
    """
    enumeration type and function for socket messages
    """
    BBOX, TARGET_NOT_FOUND, TARGET_LOST, START, STOP, TERMINATE = (i for i in range(6))


def serialize_to_msg(msg_type, bbox=None):
    if bbox is None:
        return struct.pack('!H8x', msg_type)
    return struct.pack('!Hhhhh', msg_type, *bbox.tf_rect())


def parse_from_msg(msg):
    return struct.unpack('!H', msg)[0]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help='Path to optional input video file')
    parser.add_argument('-o', '--output', help='Path to optional output video file')
    parser.add_argument('-m', '--mot', action='store_true', help='Turn on multi-object tracking')
    parser.add_argument('-s', '--socket', action='store_true', help='Turn on socket communication')
    parser.add_argument('--addr', default='/tmp/guardian_socket', help='Socket address')
    parser.add_argument('-l', '--log', action='store_true', help='Output a MOT format tracking log')
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

    delay = 0
    # Hack: delay camera frame grabbing to reduce lag
    if args['input'] is None:
        if args['mot']:
            delay = 1 / 30 # main processing loop time
        if args['gui']:
            delay += 0.025 if args['mot'] else 0.055 # gui time
    stream = VideoIO(PROC_SIZE, args['input'], args['output'], delay)

    sock = None
    mot_log = None
    enable_analytics = False
    elapsed_time = 0    
    gui_time = 0

    if args['mot']:
        analytics = Analytics(PROC_SIZE, stream.capture_dt, args['gui'] or args['output'])
        enable_analytics = True
    if args['socket']:
        assert args['mot'], 'Tracking must be turned on for socket transfer'
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args['addr'])
        sock.setblocking(False)
        enable_analytics = False
        buffer = b''
    if args['log']:
        assert args['mot'], 'Tracking must be turned on for logging'
        mot_log = open('mot_log.txt', 'w')
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
                    buffer += sock.recv(MSG_LENGTH - len(buffer))
                except OSError as err:
                    if err.args[0] != errno.EAGAIN and err.args[0] != errno.EWOULDBLOCK:
                        raise
                else:
                    if len(buffer) == MSG_LENGTH:
                        signal = parse_from_msg(buffer)
                        buffer = b''
                        if signal == MsgType.START:
                            print('client: start')
                            if not enable_analytics:
                                analytics.reset()
                                elapsed_time = 0
                                enable_analytics = True
                        elif signal == MsgType.STOP:
                            print('client: stop')
                            if enable_analytics:
                                enable_analytics = False
                                avg_fps = round(analytics.frame_count / elapsed_time)
                                print('[INFO] Average FPS: %d' % avg_fps)
                        elif signal == MsgType.TERMINATE:
                            print('client: terminate')
                            stream.stop_capture()
                            break

            if enable_analytics:
                analytics.run(frame)
                if args['log']:
                    for track_id, track in analytics.tracker.tracks.items():
                        scaled_xmin = track.bbox.xmin / PROC_SIZE[0] * stream.vid_size[0]
                        scaled_ymin = track.bbox.ymin / PROC_SIZE[1] * stream.vid_size[1]
                        scaled_xmax = track.bbox.xmax / PROC_SIZE[0] * stream.vid_size[0]
                        scaled_ymax = track.bbox.ymax / PROC_SIZE[1] * stream.vid_size[1]
                        mot_log.write(f'{analytics.frame_count + 1}, {track_id + 1}, {scaled_xmin}, {scaled_ymin}, {scaled_xmax - scaled_xmin + 1}, {scaled_ymax - scaled_ymin + 1}, -1, -1, -1, -1\n')
                if args['socket']:
                    if analytics.status == Analytics.Status.TARGET_ACQUIRED:
                        msg = serialize_to_msg(MsgType.BBOX, analytics.get_target_bbox())
                        sock.sendall(msg)
                    elif analytics.status == Analytics.Status.TARGET_NOT_FOUND:
                        msg = serialize_to_msg(MsgType.TARGET_NOT_FOUND)
                        sock.sendall(msg)
                    elif analytics.status == Analytics.Status.TARGET_LOST:
                        msg = serialize_to_msg(MsgType.TARGET_LOST)
                        sock.sendall(msg)

            if args['gui']:
                tic2 = time.perf_counter()
                # cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stream.stop_capture()
                    break
                toc2 = time.perf_counter()
                gui_time += toc2 - tic2
            if args['output'] is not None:
                stream.write(frame)
            
            toc = time.perf_counter()
            elapsed_time += toc - tic
    finally:
        # clean up resources
        stream.release()
        if sock is not None:
            sock.close()
        if mot_log is not None:
            mot_log.close()
        cv2.destroyAllWindows()
    
    if not args['socket'] and args['mot']:
        avg_fps = round(analytics.frame_count / elapsed_time)
        print('[INFO] Average FPS: %d' % avg_fps)
        if args['gui']:
            avg_time = gui_time / analytics.frame_count
            print('[INFO] Average GUI time: %f' % avg_time)


if __name__ == '__main__':
    main()
