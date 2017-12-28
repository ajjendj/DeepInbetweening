#!/usr/bin/env python

import argparse
import cv2
import os


#------------------------------------------------------------------------------------------------
def convert_vid2frames(video, output_dir,step_size=1,resize=False,new_width=None,new_height=None):
    '''
    Inputs a video, converts it into jpeg frames, resizes it with given width,
    height parameters and saves the frames to output_dir

    '''
    #Persistent variables
    vidcap = cv2.VideoCapture(video)
    video_name = os.path.splitext(os.path.basename(video))[0]

    #Variables set for every new frame
    count = 0
    success, frame = vidcap.read()

    while (success):
        if resize:
            frame = cv2.resize(frame, (new_width, new_height),interpolation=cv2.INTER_AREA)
        for i in range(step_size):
            success = vidcap.grab()

        ofile = os.path.join(output_dir,'%s_%07d.jpg' % (video_name, count))
        cv2.imwrite(ofile,frame)
        count += 1
        success, frame = vidcap.read()

    vidcap.release()
    return

#------------------------
if __name__ == "__main__":
    '''
    Converts a directory containing videos into frames and saves
    them into a directory.
    '''

    parser = argparse.ArgumentParser(description='Get video frames')
    parser.add_argument(
        '--video_dir',action='store',type=str,required=True)
    parser.add_argument(
        '--output_dir', action='store',type=str, default='/tmp')
    args = parser.parse_args()

    #state/settings
    step_size = 1
    resize = True
    new_width = 252
    new_height = 136

    #get all files in video_dir
    file_paths = [os.path.join(root,filename) for root,directories,files in os.walk(args.video_dir) for filename in files]

    #filter only .mp4 and .avi files
    video_list = [f for f in file_paths if (f.endswith(".mp4") or f.endswith(".avi"))]

    #for each video, convert video 2 frames (according to resize params) and save in output_dir
    video_count = 1
    for video in video_list:
        video_name = os.path.splitext(os.path.basename(video))[0]
        print "Converting ", video_name, " to frames.", video_count,"/",len(video_list)
        convert_vid2frames(video, args.output_dir, step_size, resize, new_width, new_height)
        print "Successful conversion of %s!" % (video_name)
        video_count += 1
