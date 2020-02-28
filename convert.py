"""
Convert jpg frames to mp4 video
"""

from pathlib import Path
import tqdm
import argparse
import os
import cv2


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', help='Path to video frame directory')
    parser.add_argument('-o', '--output', help='Path to output video file')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    args = vars(parser.parse_args())

    frame_paths = sorted(Path(args['input']).glob('*.jpg'), key=lambda path: path.name)
    first_frame = cv2.imread(str(frame_paths[0]))
    size = first_frame.shape[:2][::-1]

    Path(args['output']).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args["output"], fourcc, args['fps'], size, isColor=True)

    for frame_path in tqdm.tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    writer.release()
    os.system('rm ' + str(Path(args['input']) / '*.jpg'))
    

if __name__ == '__main__':
    main()
