#!/usr/bin/env python

import cv2
import numpy as np
import random
import scipy.misc

import util.img_util as img_util

def get_new_height(shape, new_width):
    aspect_ratio = float(shape[0]) / shape[1]  # H / W
    new_height = int(new_width * aspect_ratio)  # H' = W' / W * H
    return new_height


__processing_width = 300
def normalize_for_analysis(frame, width=__processing_width):
    processing_height = get_new_height(np.shape(frame), __processing_width)
    res = cv2.resize(frame, (__processing_width, processing_height))
    if len(np.shape(frame)) > 2:
        # Detected 3 channels
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


class MotionAnalysis(object):
    '''
    Analyzes and saves motion information for a pair of frames.
    '''
    def __init__(self, f0, f1):
        self.diff = cv2.absdiff(f1, f0)
        self.shape = np.shape(f1)
        self.pixels = self.shape[0] * self.shape[1]


    def likely_frame_cut(self):
        # TODO: collect data; evaluate; tune this detector
        return self.diff.max() > 150 and self.diff.mean() > 20


    def likely_same_frame(self):
        # TODO: collect data; evaluate; tune this detector
        return self.diff.max() < 100 and self.diff.mean() < 2


    def motion_metric(self):
        # TODO: need a more robust metric
        return self.diff.mean()



class MotionLocalizer(object):
    '''
    Localizes regions of motion for frames f0, f1, f2.
    '''
    def __init__(self, frames_norm, motion_analysis0, motion_analysis1):
        self.frames = frames_norm
        self.analysis0 = motion_analysis0
        self.analysis1 = motion_analysis1
        self.rects = []


    def motion_rectangles(self):
        return self.rects


    def stochastic_localization(self, loc_width=20, box_range=(2,5)):
        # Compute difference between f0 and f2
        diff = cv2.absdiff(self.frames[0], self.frames[-1])

        # Remove obvious near zero pixels
        th0 = max(5, np.median(diff))
        th = cv2.threshold(diff, th0, 255, cv2.THRESH_TOZERO)[1]
        # scipy.misc.imsave('/tmp/th.png', th)

        th = cv2.dilate(th, None, iterations=2)
        # scipy.misc.imsave('/tmp/dil.png', th)

        # Change to tiny size
        loc_width = 30
        loc_height = get_new_height(np.shape(th), loc_width)

        th = scipy.misc.imresize(th, (loc_height, loc_width), interp='nearest')
        # scipy.misc.imsave('/tmp/th3.png', th)

        # Keep the top 25%
        th1 = np.percentile(th[np.nonzero(th)], 75)
        th_top75 = cv2.threshold(th, th1, 255, cv2.THRESH_TOZERO)[1]
        scipy.misc.imsave('/tmp/th_top.png', th_top75)
        unaccounted = np.count_nonzero(th_top75)

        while unaccounted > 0:
            print 'Unaccounted %d ' % unaccounted
            # get random high-intensity pixel
            elems = np.nonzero(th_top75)
            idx = random.randint(0, len(elems[0]) - 1)
            y = elems[0][idx]
            x = elems[1][idx]

            # create randomish box size around it
            # Maybe randomness determined by the neighborhood?
            # Maybe create 3 random ones and pick the best one?
            # (E.g. we want some context for this)
            width = random.randint(box_range[0], box_range[1])
            rect = img_util.ImageRect.create_from_center(
                (x,y), width, width, np.shape(th), accept_degenerate=True)
            rect.make_square()  # If degenerate square clipped to rectangle
            self.rects.append(rect)

            # Now black out the box in th_top75
            rect.set_subarray(th_top75, 0)
            unaccounted = np.count_nonzero(th_top75)
            # What happens if we expand this? Does motion mean stay the same?
            #np.where(th[1] > 100)


    def analyze_bw_contours(self):
        # TODO(ajjen): if necessary, put b/w motion localization code here
        pass
