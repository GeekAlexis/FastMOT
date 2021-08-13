#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import json
import cv2

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                        'URI to input stream\n'
                        '1) image sequence (e.g. %%06d.jpg)\n'
                        '2) video file (e.g. file.mp4)\n'
                        '3) MIPI CSI camera (e.g. csi://0)\n'
                        '4) USB camera (e.g. /dev/video0)\n'
                        '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                        '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    parser.add_argument('-c', '--config', metavar="FILE",
                        default=Path(__file__).parent / 'cfg' / 'mot.json',
                        help='path to JSON configuration file')
    parser.add_argument('-l', '--labels', metavar="FILE",
                        help='path to label names (e.g. coco.names)')
    parser.add_argument('-o', '--output_uri', metavar="URI",
                        help='URI to output video file')
    parser.add_argument('-t', '--txt', metavar="FILE",
                        help='output MOT Challenge txt results (e.g. eval/results/MOT20-01.txt)')
    parser.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    parser.add_argument('-g', '--gui', action='store_true', help='enable display')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)

    stream = fastmot.VideoIO(config.resize_to, args.input_uri, args.output_uri, **vars(config.stream_cfg))

    mot = None
    txt = None
    if args.mot:
        draw = args.gui or args.output_uri is not None
        mot = fastmot.MOT(config.resize_to, **vars(config.mot_cfg), draw=draw)
        mot.reset(stream.cap_dt)
        if args.txt is not None:
            assert Path(args.txt).suffix == '.txt'
            Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
            txt = open(args.txt, 'w')
    if args.gui:
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        with Profiler('app') as prof:
            while not args.gui or cv2.getWindowProperty('Video', 0) >= 0:
                frame = stream.read()
                if frame is None:
                    break

                if args.mot:
                    mot.step(frame)
                    if txt is not None:
                        for track in mot.visible_tracks():
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                      f'{w:.6f},{h:.6f},-1,-1,-1\n')

                if args.gui:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # timing statistics
    if args.mot:
        avg_fps = round(mot.frame_count / prof.duration)
        logger.info('Average FPS: %d', avg_fps)
        mot.print_timing_info()


if __name__ == '__main__':
    main()
