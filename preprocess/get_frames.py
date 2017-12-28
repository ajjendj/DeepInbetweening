#!/usr/bin/env python

import argparse
import cv2
import os

from util.basics import *

if __name__ == "__main__":
    '''
    Converts a video into frames.
    Must run as module with python -m.
    '''

    parser = argparse.ArgumentParser(description='Get Video Frames')
    parser.add_argument(
        '--video', action='store', type=str, required=True)
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True)
    parser.add_argument(
        '--frame_file_pattern', action='store',
        type=str, default='frame%07d.jpg')
    args = parser.parse_args()

    vidcap = cv2.VideoCapture(args.video)

    count = 0;
    while True:
        success,image = vidcap.read()
        if not success:
            break
        filename = os.path.join(args.output_dir,
                                args.frame_file_pattern % count)
        if not cv2.imwrite(filename, image):
            print_error('Failed to write %s' % filename)
            break
        count += 1
