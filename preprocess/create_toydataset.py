#usr/bin/env python

import argparse
import cv2
import numpy as np
import os

#------------------------
if __name__ == "__main__":
    '''
    Creates a toy dataset of transformed shapes.
    *Take1: Ellipses undergoing translation only in the x-direction
    *Take2: Ellipses undergoing translation in x and y directions
    '''

    parser = argparse.ArgumentParser(description='Get dir to store files')
    parser.add_argument(
        '--output_dir', action='store',type=str, default='/tmp')
    args = parser.parse_args()

    width = 32
    height = 32
    num_samples = 32

    #for each sample, create a motion_triplet 3-channel image of the following:
    #   an ellipse drawn on the first channel
    #   the ellipse translated in the x-direction in the second channel
    #   the difference image in the third channel

    num = 0
    while num < num_samples:
        #X, y = create_shape_transforms(width, height, args.output_dir)
        img_A = np.zeros((width,height,3), np.uint8)
        img_B = np.zeros((width,height,3), np.uint8)
        img_C = np.zeros((width,height,3), np.uint8)
        cx = np.random.randint(10,width-10)
        cy = np.random.randint(10,height-10)
        majax = np.random.randint(5,10)
        minax = np.random.randint(5,10)
        angle = np.random.randint(0,12) * 30
        trans_x = np.random.randint(-5,5)
        trans_y = np.random.randint(-5,5)
        rot = np.random.randint(0,90)
        scl_maj = np.random.randint(-2,2)
        scl_min = np.random.randint(-2,2)
        col_a = np.random.randint(2)
        col_b = np.random.randint(2)
        col_c = np.random.randint(2)
        col_list = [128,255]
        if (((2*trans_x - (majax + 2*scl_maj)) >= 0) and
            ((2*trans_x + (majax + 2*scl_maj)) < width) and
            ((2*trans_y - (minax + 2*scl_min)) >= 0) and
            ((2*trans_y + (majax + 2*scl_min)) < width)):
            num += 1
            cv2.ellipse(img_A, (cx, cy), (majax,minax),
                angle, 0, 360, (col_list[col_a], col_list[col_b], col_list[col_c]), -1)
            cv2.ellipse(img_B, (cx + trans_x, cy + trans_y), (majax + scl_maj, minax + scl_min),
                angle + rot, 0, 360, (col_list[col_a], col_list[col_b], col_list[col_c]), -1)
            cv2.ellipse(img_C, (cx + 2*trans_x, cy + 2*trans_y), (majax + 2*scl_maj,minax + 2*scl_min),
                angle + 2*rot, 0, 360, (col_list[col_a], col_list[col_b], col_list[col_c]), -1)
            cv2.imwrite(args.output_dir + "/" + "TRS" +"%07dA.png" %(num),img_A)
            cv2.imwrite(args.output_dir + "/" + "TRS" +"%07dB.png" %(num),img_B)
            cv2.imwrite(args.output_dir + "/" + "TRS" +"%07dC.png" %(num),img_C)
            #np.savetxt(args.output_dir + "/" + "translation" +"%07d.csv" %(num),np.asarray([1, 0, trans_x, 0, 1, trans_y]))
    print "Successful creation of %d toy transformation triplets." % num_samples

