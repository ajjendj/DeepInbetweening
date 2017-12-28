#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os


#------------------------
def motion_triplets(video, output_dir, step_size=1, resize=False, new_width=None, new_height=None, thresh=False):
    '''
    Inputs a video, converts it into individual frames, localizes
    bounding boxes of motion between triplets of frames (spacing
    determined by step_size parameter) and saves appropriate
    frame-triplets as 3-channel images to be used for training.
    '''

    #Persistent variables
    vidcap = cv2.VideoCapture(video)
    video_name = os.path.splitext(os.path.basename(video))[0]

    #Create sub-directory in which to save motion_triplet images
    os.mkdir(output_dir+"/"+video_name)

    #Get total number of frames
    tot_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Get frame rate
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))
    #debug: print tot_frames, frame_rate

    #Throw away first 6 seconds (corresponding to opening credits)
    for i in range(6 * frame_rate):
        s = vidcap.grab()

    #Variable set for every new saved image
    write_count = 0

    #Variables set for every new triplet
    frame_count = 6 * frame_rate
    s0, frame0 = vidcap.read()

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    if thresh:
        _,frame0 = cv2.threshold(frame0, 127, 255, cv2.THRESH_BINARY)
    for i in range(1,step_size):
        s = vidcap.grab()
        frame_count += 1
    s1, frame1 = vidcap.read()
    frame_count += 1
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if thresh:
        _,frame1 = cv2.threshold(frame1, 127, 255, cv2.THRESH_BINARY)
    for i in range(1,step_size):
        s = vidcap.grab()
        frame_count += 1
    s2, frame2 = vidcap.read()
    frame_count += 1
    #placeholder for output composite frame
    frame_out = frame2.copy()
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    if thresh:
        _,frame2 = cv2.threshold(frame2, 127, 255, cv2.THRESH_BINARY)
    for i in range(1,step_size):
        s = vidcap.grab()
        frame_count += 1

    #Read until t-12 seconds of video (corresponding to display of subscription ads)
    while (s2 and (frame_count < (tot_frames - frame_rate * 12))):

        #output frame is composite of the 3 grayscale images
        frame_out[:,:,0] = frame0
        frame_out[:,:,1] = frame1
        frame_out[:,:,2] = frame2

        #get motion history image (aggregation of the absdiffed images)
        frame_diff0 = cv2.absdiff(frame1, frame0)
        frame_diff1 = cv2.absdiff(frame2, frame1)
        frame_diff01 = cv2.absdiff(frame_diff0, frame_diff1)

        #dilate the motion history image
        motion_history = cv2.dilate(frame_diff01, None, iterations=2)

        #debug: cv2.imshow('motionhistory', motion_history)

        #find contours
        _, contours, hierarchy = cv2.findContours(motion_history, 0,2)

        #threshold between the layers of the output composite image
        #    basically prevents any pair of the 3 images being different
        absdiff_thresh = 40000
        frame_out_rect = frame_out.copy()

        #for all detected blobs in the motion history image
        for c in contours:

            #discard small contours i.e. areas with very small changes in motion
            if cv2.contourArea(c) < 1000:
                continue

            #get bounding rectangle of contour and crop and scale a 128x128 image
            #    from the composite output centered in the center of the bounding rectangle
            (x, y, w, h) = cv2.boundingRect(c)
            dim = h
            if w > h:
                dim = w
            cx = x + w/2
            cy = y + h/2
            crop_out = frame_out[cy - dim/2:cy + dim/2, cx - dim/2: cx + dim/2]

            #check if crop_out is valid
            if (np.shape(crop_out)[0] == 0) or (np.shape(crop_out)[1] == 0):
                continue
            np.shape(crop_out)
            crop_out = cv2.resize(crop_out, (new_width,new_height), interpolation=cv2.INTER_AREA)

            #save output if crop_out represents a good candidate for training
            if (np.sum(cv2.absdiff(crop_out[:,:,0], crop_out[:,:,1])) > absdiff_thresh and np.sum(cv2.absdiff(crop_out[:,:,1], crop_out[:,:,2])) > absdiff_thresh and np.sum(cv2.absdiff(crop_out[:,:,2], crop_out[:,:,0])) > absdiff_thresh):
                cv2.imwrite(output_dir + "/" + video_name + "/" +"%s_%07d.png" %(video_name, write_count),crop_out)
                #if write_count == 164 or write_count == 171 or write_count == 259:
                #    cv2.imwrite("ttt1" +"%s_%07d.jpg" %(video_name, write_count),crop_out[:,:,0])
                #    cv2.imwrite("ttt2" +"%s_%07d.jpg" %(video_name, write_count),crop_out[:,:,1])
                #    cv2.imwrite("ttt3" +"%s_%07d.jpg" %(video_name, write_count),crop_out[:,:,2])
                write_count += 1
                #debug: cv2.rectangle(frame_out_rect, (cx - dim/2, cy - dim/2), (cx + dim/2, cy + dim/2), (0, 255, 0), 2)

        #Prepare for next frame triplet
        frame0 = frame1
        frame1 = frame2
        s2, frame2 = vidcap.read()
        frame_count += 1
        frame_out = frame2.copy()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if thresh:
            _,frame2 = cv2.threshold(frame2, 127, 255, cv2.THRESH_BINARY)
        for i in range(1,step_size):
            s = vidcap.grab()
            frame_count += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vidcap.release()

    return write_count

#------------------------
if __name__ == "__main__":
    '''
    Converts a directory containing videos into frame-triplets
    to be used in training a deep network that predicts the
    middle frame.
    '''

    parser = argparse.ArgumentParser(description='Get video')
    parser.add_argument(
        '--video_dir',action='store',type=str,required=True)
    parser.add_argument(
        '--output_dir', action='store',type=str, default='/tmp')
    args = parser.parse_args()

    #state/settings
    step_size = 1
    resize = True
    new_width = 48
    new_height = 48

    #get all files in video dir
    file_paths = [os.path.join(root,filename) for root,directories,files in os.walk(args.video_dir) for filename in files]

    #filter only .mp4 and .avi files
    video_list = [f for f in file_paths if (f.endswith(".mp4") or f.endswith(".avi"))]

    #for each video, create motion_triplet representations and save in output_dir
    video_count = 1
    for video in video_list:
        video_name = os.path.splitext(os.path.basename(video))[0]
        print "Converting ", video_name, "to frames.", video_count,"/",len(video_list)
        num_mt = motion_triplets(video, args.output_dir, step_size, resize, new_width, new_height)
        print "Successful creation of %d motion triplets." % num_mt
        video_count +=1
