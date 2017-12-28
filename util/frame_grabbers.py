#!/usr/bin/env python

from abc import ABCMeta, abstractmethod

import cv2
import scipy.misc


class FrameGrabber(object):
    '''
    Abstraction for frame-grabbing mechanism that allows plugging in
    video and images into the same algorithm.
    '''
    __metaclass__ = ABCMeta
    BGR_FORMAT = 'BGR'
    RGB_FORMAT = 'RGB'

    def __init__(self, channel_format):
        if channel_format not in [FrameGrabber.BGR_FORMAT, FrameGrabber.RGB_FORMAT]:
            raise RuntimeError('Unknown channel format %s' % channel_format)
        self.channel_format = channel_format


    def skip_seconds(self, seconds_to_skip):
        ''' Skips specified seconds forward, if there is frame rate. '''
        frame_count = 0
        frames_to_skip = seconds_to_skip * self.frame_rate()
        print 'Skipping %d frames' % frames_to_skip  # E.g. opening credits
        for i in range(frames_to_skip):
            self.skip_frame()
            frame_count += 1
        return frame_count


    @abstractmethod
    def skip_frame(self):
        pass


    @abstractmethod
    def next_frame(self):
        pass


    @abstractmethod
    def frame_rate(self):
        pass


    @abstractmethod
    def total_frames(self):
        pass



class VideoFrameGrabber(FrameGrabber):
    '''
    Grabs frame from a video stream.
    '''
    def __init__(self, video_file, channel_format=FrameGrabber.BGR_FORMAT):
        super(VideoFrameGrabber, self).__init__(channel_format=channel_format)
        self.vidcap = cv2.VideoCapture(video_file)
        self.n_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))


    def skip_frame(self):
        self.vidcap.grab()


    def next_frame(self):
        success, frame = self.vidcap.read()
        if not success:
            return None
        # OpenCV is BGR by default
        if self.channel_format == FrameGrabber.RGB_FORMAT:
            frame = cv.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


    def frame_rate(self):
        return self.fps


    def total_frames(self):
        return self.n_frames



class FileFrameGrabber(FrameGrabber):
    '''
    Grabs frames in order from a given set of files.
    '''
    def __init__(self, files, channel_format=FrameGrabber.BGR_FORMAT):
        super(FileFrameGrabber, self).__init__(channel_format=channel_format)
        self.files = files
        self.idx = 0


    def skip_frame(self):
        self.idx += 1


    def next_frame(self):
        if self.idx >= len(self.files):
            return None
        res = scipy.misc.imread(self.files[self.idx])
        self.idx += 1
        # Scipy is RGB by default
        if self.channel_format == FrameGrabber.BGR_FORMAT:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        return res


    def frame_rate(self):
        return 0


    def total_frames(self):
        return len(self.files)
