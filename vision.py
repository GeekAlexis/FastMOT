import argparse
import time
import os
import socket
import errno
import cv2

from analytics import VideoIO
from analytics import Analytics


"""
constants
"""
MSG_LENGTH = 16
PASSWORD = 'jdyw123'
PROC_SIZE = (1280, 720)


class Msg:
    """
    enumeration type and function for socket messages
    """
    TARGET_NOT_FOUND, TARGET_LOST, START, STOP, TERMINATE = ((i).to_bytes(MSG_LENGTH, byteorder='big') for i in range(5))


def convert_bbox_to_bytes(bbox):
    length = MSG_LENGTH // 4
    return b''.join(int(coord).to_bytes(length, byteorder='big', signed=True) for coord in bbox.tf_rect())


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
            delay = 1 / 30 # main processing loop time
        if args['gui']:
            delay += 0.025 if args['analytics'] else 0.055 # gui time
    stream = VideoIO(PROC_SIZE, args['input'], args['output'], delay)

    sock = None
    enable_analytics = False
    elapsed_time = 0    
    gui_time = 0

    if args['analytics']:
        analytics = Analytics(PROC_SIZE, stream.capture_dt, args['gui'] or args['output'])
        enable_analytics = True
    if args['socket']:
        assert args['analytics'], 'Analytics must be turned on for socket transfer'
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
                            analytics.reset()
                            elapsed_time = 0
                            enable_analytics = True
                    elif msg == Msg.STOP:
                        print('client: stop')
                        if enable_analytics:
                            enable_analytics = False
                            avg_fps = round(analytics.frame_count / elapsed_time)
                            print('[INFO] Average FPS: %d' % avg_fps)
                    elif msg == Msg.TERMINATE:
                        print('client: terminate')
                        stream.stop_capture()
                        break

            if enable_analytics:
                analytics.run(frame)
                if args['socket']:
                    if analytics.status == Analytics.Status.TARGET_ACQUIRED:
                        sock.sendall(convert_bbox_to_bytes(analytics.get_target_bbox()))
                    elif analytics.status == Analytics.Status.TARGET_NOT_FOUND:
                        sock.sendall(Msg.TARGET_NOT_FOUND)
                    elif analytics.status == Analytics.Status.TARGET_LOST:
                        sock.sendall(Msg.TARGET_LOST)

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
        cv2.destroyAllWindows()
    
    if not args['socket']:
        avg_fps = round(analytics.frame_count / elapsed_time)
        print('[INFO] Average FPS: %d' % avg_fps)
        if args['gui']:
            avg_time = gui_time / analytics.frame_count
            print('[INFO] Average GUI time: %f' % avg_time)


if __name__ == '__main__':
    main()
