#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import scipy.misc

import util.frame_grabbers as fg
import preprocess.motion_analysis as ma
import preprocess.synthetic_transforms as st


def binary_threshold_frame(frame):
    # TODO: make configurable if needed
    _, res = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    return res


def to_grayscale(frame):
    # TODO: is it always BGR?
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


FRAME_PREPROCESSORS = {
    'to_gray': to_grayscale,
    'binary_thresh': binary_threshold_frame
    }

SYNTHETIC_TRANSFORMS = [
    'AFFINE', 'TRANSLATE'
    ]

def parse_preprocessors(preps):
    preprocessors = []
    if len(preps) == 0:
        return preprocessors
    name_list = preps.split(',')
    for prep_name in name_list:
        if prep_name not in FRAME_PREPROCESSORS:
            raise RuntimeError('Preprocessor "%s" not known' % prep_name)
        else:
            print '   * using preprocessor %s' % prep_name
            preprocessors.append(FRAME_PREPROCESSORS[prep_name])
    return preprocessors

def set_up_frame_grabber(args):
    frame_grabber = None
    output_prefix = None
    if args.video is not None:
        frame_grabber = fg.VideoFrameGrabber(args.video)
        file_prefix = args.output_prefix
        if file_prefix is None:
            file_prefix = os.path.splitext(os.path.basename(args.video))[0]
        output_prefix = os.path.join(args.output_dir, file_prefix)
        print ('Grabbing %d frames from %s to %s' %
               (frame_grabber.total_frames(), args.video, output_prefix))
    else:
        frame_grabber = fg.FileFrameGrabber(args.frames.split(','))
        file_prefix = args.output_prefix
        if file_prefix is None:
            file_prefix = 'train_frame'
        output_prefix = os.path.join(args.output_dir, file_prefix)
        print ('Grabbing %d frames from input images to %s' %
               (frame_grabber.total_frames(), output_prefix))

    return frame_grabber, output_prefix


if __name__ == "__main__":
    '''
    Converts a video or a csv list of frames into training data.

    Operates in two modes:
    1. True motion: for each triplet of frames with sufficient motion,
       outputs 3 images of image_size x image_size, corresponding to
       a region that had motion in the triplet
    2. Synthetic motion: for each frame and regions in it that had
       some motion, create a synthetic affine transform of that region
       and output region, partially warped region, and fully warped region.
       Activate this mode with --synthetic_motion=AFFINE or
       --synthetic_motion=TRANSLATE

    Output format:
    TODO(shumash): adjust to conform with Ajjen's loaders and document
    '''

    parser = argparse.ArgumentParser(description='Turn video into training data')
    parser.add_argument(
        '--video', action='store', type=str, required=False)
    parser.add_argument(
        '--frames', action='store', type=str, required=False)
    parser.add_argument(
        '--output_prefix', action='store', type=str, required=False)
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True)
    parser.add_argument(
        '--output_extension', action='store', type=str, default='png')
    parser.add_argument(
        '--image_size', action='store', type=int, default=48)
    parser.add_argument(
        '--seconds_to_skip_start', action='store', type=int, default=6)
    parser.add_argument(
        '--seconds_to_skip_end', action='store', type=int, default=12)
    parser.add_argument(
        '--frames_to_skip', action='store', type=int, default=np.random.randint(0,1))
    parser.add_argument(
        '--preprocessors', action='store', type=str, default='')
    parser.add_argument(
        '--min_motion', action='store', type=float, default = 0.1)
    parser.add_argument(
        '--synthetic_motion', action='store', type=str, required=False,
        help='If TRANSLATE or AFFINE, generates synthetic motion data instead')
    args = parser.parse_args()

    frames_to_skip = args.frames_to_skip

    if (args.video is not None) == (args.frames is not None):
        raise RuntimeError('Must set exactly one of --video and --frames')

    if args.synthetic_motion is not None:
        if args.synthetic_motion not in SYNTHETIC_TRANSFORMS:
            raise RuntimeError('Unknown synthetic transform %s' % args.synthetic_motion)
        print 'Generating synthetic data with %s transforms' % args.synthetic_motion

    # Set up frame grabbing from a set of frames or video
    frame_grabber, output_prefix = set_up_frame_grabber(args)

    # Skip frames at the beginning if specified
    frame_count = 0
    frame_count = frame_grabber.skip_seconds(args.seconds_to_skip_start)
    last_frame = (frame_grabber.total_frames() -
                  args.seconds_to_skip_end * frame_grabber.frame_rate())

    print 'Parsing preprocessors...'
    preprocessors = parse_preprocessors(args.preprocessors)

    def get_next_frame():
        ''' Grabs next frame and runs preprocessors. '''
        print 'F %d' % frame_count
        frame = frame_grabber.next_frame()
        frame_norm = None
        if frame is not None:
            for prep in preprocessors:
                frame = prep(frame)
            frame_norm = ma.normalize_for_analysis(frame)
        return frame, frame_norm

    def skip_frames(frames_to_skip):
        ''' Skips the specified number of frames. '''
        for i in range(frames_to_skip):
            frame = frame_grabber.next_frame()
        return

    triplet_num = 0
    rect_num = 0
    def get_output_name(triplet_image_id):
        '''
        Gets output name for current triplet_num, the number of motion
        rectangle in that frame, as well as the frame id inside the
        output motion triplet.
        '''
        return ('%s_%010d_%03d_%s.%s' %
                (output_prefix, triplet_num, rect_num, triplet_image_id,
                 args.output_extension))


    def output_motion(rect, frames):
        '''
        In standard mode, simply outputs the crop of rect in all 3 frames
        resized to image_size x image_size.
        In synthetic motion mode, modifies frame0 only with a random synthetic
        transform and crops out the motion triplet from frame0 and warped frames.
        '''
        output_frames = frames
        shape = np.shape(frames[0])
        if args.synthetic_motion is not None:
            transformer = st.Transformer(rect)
            M = None
            if args.synthetic_motion == 'TRANSLATE':
                trans = transformer.random_translation()
                M = transformer.translation_matrix(trans, homogeneous=False)
                M_half = transformer.translation_matrix(trans * 0.5, homogeneous=False)
            elif args.synthetic_motion == 'AFFINE':
                M, M_half = transformer.random_affine_transform(homogeneous=False)
            else:
                raise RuntimeError('Unknown synthetic motion type %s' % args.synthetic_motion)
            f0 = frames[0]
            f1 = cv2.warpAffine(f0, M_half, (shape[1], shape[0]), borderValue=(0,0,0))
            f2 = cv2.warpAffine(f0, M, (shape[1], shape[0]), borderValue=(0,0,0))
            output_frames = [f0, f1, f2]
            # Optionally output numeric transform here
        framename_id = ['A', 'B', 'C']
        for i in range(0,3):
            crop = cv2.resize(rect.subarray(output_frames[i]),
                              (args.image_size, args.image_size))
            if len(np.shape(crop)) == 3:  # color image
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # scipy writes in RGB
            scipy.misc.imsave(get_output_name(framename_id[i]), crop)


    frame, frame_norm = get_next_frame()
    frame_count += 1

    skip_frames(frames_to_skip)
    frame_count += frames_to_skip

    if frame is None:
        raise RuntimeError('Could not read first frame')

    # Accumulate 3 frames and their analysis before outputting
    frames = [ frame ]
    frames_norm = [ frame_norm ]
    analysis = []
    while frame_count < last_frame:  # optionally skip a few frames at the end
        if len(frames) < 3:
            frame, frame_norm = get_next_frame()
            frame_count += 1

            if frame is None:
                print 'Finished processing at frame %d' % frame_count
                break

            if len(frames_norm) == 0:  # if all frames removed at frame cut
                frames.append(frame)
                frames_norm.append(frame_norm)
            else:
                mot = ma.MotionAnalysis(frames_norm[-1], frame_norm)
                if mot.likely_frame_cut():
                    print 'Detected frame cut'
                    frames = []
                    frames_norm = []
                    analysis = []
                elif (not mot.likely_same_frame() and
                      mot.motion_metric() > args.min_motion):
                    frames.append(frame)
                    frames_norm.append(frame_norm)
                    analysis.append(mot)
                else:
                    print 'Skipping frame, insufficient motion'
        if len(frames) == 3:
            full_shape = np.shape(frames[0])
            localizer = ma.MotionLocalizer(frames_norm, analysis[0], analysis[1])
            localizer.stochastic_localization()
            rects = localizer.motion_rectangles()
            rect_num = 0  # for naming
            for rect in rects:
                rect.make_square()
                rect.resize_for_image(full_shape)
                output_motion(rect, frames)
                rect_num += 1
            triplet_num += 1
            frames.pop(0)
            frames_norm.pop(0)
            analysis.pop(0)

            if np.random.uniform() > 0.5: #with probability 0.5 skip 'frames_to_skip' number of frames
                frames = []
                frames_norm = []
                analysis = []
                skip_frames(frames_to_skip)
                frame_count += frames_to_skip

