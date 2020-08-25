#!/usr/bin/env python3
import argparse
import logging
import socket
import struct
import errno
import time
import cv2

from fast_mot import VideoIO
from fast_mot import Mot


"""
constants
"""
MSG_LENGTH = 2  # 2-byte short
PROC_SIZE = (1280, 720)


class MsgType:
    """
    enumeration type and function for socket messages
    """
    START, STOP, TERMINATE = (i for i in range(3))


def serialize_to_msg(frame_num, trk_id, tl, w, h):
    return struct.pack('!NNhhhh', frame_num, trk_id, *tl, w, h)


def parse_from_msg(msg):
    return struct.unpack('!H', msg)[0]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help='Path to optional input video file')
    parser.add_argument('-o', '--output', help='Path to optional output video file')
    parser.add_argument('-m', '--mot', action='store_true', help='Multi-object tracking')
    parser.add_argument('-s', '--socket', action='store_true', help='Output to a unix socket')
    parser.add_argument('--addr', default='/tmp/fast_mot_socket', help='Path to unix socket')
    parser.add_argument('-l', '--log', action='store_true', help='Output a MOT format log')
    parser.add_argument('-g', '--gui', action='store_true', help='Visiualization')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output for debugging')

    args = parser.parse_args()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

    delay = 0
    # Hack: delay camera frame grabbing to reduce lag
    if args.input is None:
        if args.mot:
            delay = 1 / 30 # main processing loop time
        if args.gui:
            delay += 0.025 if args.mot else 0.055 # gui latency
    stream = VideoIO(PROC_SIZE, args.input, args.output, delay)

    sock = None
    mot_log = None
    mot = None
    enable_mot = False
    elapsed_time = 0    
    gui_time = 0

    if args.mot:
        mot = Mot(PROC_SIZE, stream.capture_dt, args.gui or args.output, args.verbose)
        enable_mot = True
    if args.socket:
        if not args.mot:
            raise RuntimeError('There is nothing to output')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.addr)
        sock.setblocking(False)
        enable_mot = False
        buffer = b''
    if args.log:
        if not args.mot:
            raise RuntimeError('There is nothing to output')
        mot_log = open('MOT_log.txt', 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
        
    logging.info('Starting video capture...')
    stream.start_capture()
    try:
        while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
            tic = time.perf_counter()
            frame = stream.read()
            if frame is None:
                break

            if args.socket:
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
                            if not enable_mot:
                                mot.initiate()
                                elapsed_time = 0
                                enable_mot = True
                        elif signal == MsgType.STOP:
                            if enable_mot:
                                enable_mot = False
                                avg_fps = round(mot.frame_count / elapsed_time)
                                logging.info('Average FPS: %d', avg_fps)
                        elif signal == MsgType.TERMINATE:
                            stream.stop_capture()
                            break

            if enable_mot:
                mot.run(frame)
                if args.log or args.socket:
                    for trk_id, track in mot.tracker.tracks.items():
                        tl = track.bbox.tl / PROC_SIZE * stream.vid_size
                        br = track.bbox.br / PROC_SIZE * stream.vid_size
                        w, h = br - tl + 1
                        if args.log:
                            mot_log.write(f'{mot.frame_count}, {trk_id}, {tl[0]}, {tl[1]}, {w}, {h}, -1, -1, -1, -1\n')
                        else:
                            sock.sendall(serialize_to_msg(mot.frame_count, trk_id, tl, w, h))

            if args.gui:
                tic2 = time.perf_counter()
                # cv2.putText(frame, '%d FPS' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    stream.stop_capture()
                    break
                toc2 = time.perf_counter()
                gui_time += toc2 - tic2
            if args.output is not None:
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
    
    if not args.socket and args.mot:
        avg_fps = round(mot.frame_count / elapsed_time)
        avg_trk_time = mot.track_time / (mot.frame_count - mot.detector_frame_count)
        avg_embedding_time = mot.embedding_time / mot.detector_frame_count
        avg_det_pre_time = mot.det_pre_time / mot.detector_frame_count
        avg_det_time = mot.det_time / mot.detector_frame_count
        avg_match_time = mot.match_time / mot.detector_frame_count
        
        logging.info('Average FPS: %d', avg_fps)
        logging.debug('Average track time: %f', avg_trk_time)
        logging.debug('Average embedding time: %f', avg_embedding_time)
        logging.debug('Average det pre time: %f', avg_det_pre_time)
        logging.debug('Average det time: %f', avg_det_time)
        logging.debug('Average match time: %f', avg_match_time)
        if args.gui:
            avg_time = gui_time / mot.frame_count
            logging.debug('Average GUI time: %f', avg_time)


if __name__ == '__main__':
    main()
