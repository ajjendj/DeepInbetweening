#!/usr/bin/env python

import argparse
import cv2
import os

from util.basics import *

if __name__ == "__main__":
    '''
    Converts a video into frames.
    Must run as module with python -m.

    Example usage (inside virtualenv):
    python -m util.video_viewer --video=/tmp/test.mp4
    '''

    parser = argparse.ArgumentParser(description='Get Video Frames')
    parser.add_argument(
        '--video', action='store', type=str, required=True)
    parser.add_argument(
        '--output_dir', action='store', type=str, default='/tmp')
    args = parser.parse_args()

    # Persistent variables
    vidcap = cv2.VideoCapture(args.video)
    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # State/settings
    playing = False
    advance_frame = True
    speed = 1
    need_refresh = False
    autosave = False

    # Variables set for every new frame
    count = 0
    image = None
    success = True

    while (True):
        if playing or advance_frame:
            for i in range(1, speed): # For speed > 1, skip frames
                success = vidcap.grab()
                count += 1

            success, frame = vidcap.read()
            advance_frame = False
            count += 1
            need_refresh = True

            if not success:
                break

        k = cv2.waitKey(1)
        if k == ord('a') or k == ord('A'):
            autosave = not autosave
            print 'Setting frame auto-save to %s' % ('ON' if autosave else 'OFF')

        if need_refresh and (autosave or k == ord('s') or k == ord('S')):
            ofile = os.path.join(args.output_dir,
                                 '%s_%07d.jpg' % (video_name, count))
            if not cv2.imwrite(ofile, frame):
                print_error('Failed to write %s' % ofile)
            else:
                print 'Wrote %s' % ofile

        if k == ord(' '):
            playing = not playing
        elif k == ord('-'):
            speed = max(1, speed - 1)
            need_refresh = True
            print 'Set speed to: %d' % speed
        elif k == ord('+'):
            speed += 1
            need_refresh = True
            print 'Set speed to: %d' % speed
        elif k == ord('>') or k == 124:  # Right arrow
            advance_frame = True
        elif k == ord('q') or k == 27:  #ESC
            break
        elif k != -1:
            print k

        if need_refresh:
            image = frame.copy()  # put text on a copy
            message = '%07d' % count
            if speed != 1:
                message += ' at %dx speed' % speed
            cv2.putText(image, message,
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('frame', image)
            need_refresh = False

    vidcap.release()
    cv2.destroyAllWindows()
