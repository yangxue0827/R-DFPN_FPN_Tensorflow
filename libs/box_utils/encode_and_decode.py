# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from math import pi
import cv2


def angle_transform(coords):
    coords_new = []
    for coord in coords:
        points = cv2.boxPoints(((coord[0], coord[1]), (coord[2], coord[3]), coord[4]))
        coord_new = cv2.minAreaRect(points)
        coords_new.append([coord_new[0][0], coord_new[0][1], coord_new[1][0], coord_new[1][1], coord_new[2]])
    return np.array(coords_new, np.float32)


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None, name='decode'):
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w, t_theta = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]
            t_theta /= scale_factors[4]

        reference_x_center, reference_y_center, reference_w, reference_h, reference_theta = \
            tf.unstack(reference_boxes, axis=1)

        predict_x_center = t_xcenter * reference_w + reference_x_center
        predict_y_center = t_ycenter * reference_h + reference_y_center
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_theta = t_theta * 180 / pi + reference_theta

        # mask1 = tf.less(predict_theta, -90)
        # mask2 = tf.greater_equal(predict_theta, -180)
        # mask7 = tf.less(predict_theta, -180)
        # mask8 = tf.greater_equal(predict_theta, -270)
        #
        # mask3 = tf.greater_equal(predict_theta, 0)
        # mask4 = tf.less(predict_theta, 90)
        # mask5 = tf.greater_equal(predict_theta, 90)
        # mask6 = tf.less(predict_theta, 180)

        # # to keep range in [-90, 0) in almost situation
        # # [-180, -90)
        # convert_mask = tf.logical_and(mask1, mask2)
        # remain_mask = tf.logical_not(convert_mask)
        # predict_theta += tf.cast(convert_mask, tf.float32) * 90.
        #
        # remain_h = tf.cast(remain_mask, tf.float32) * predict_h
        # remain_w = tf.cast(remain_mask, tf.float32) * predict_w
        # convert_h = tf.cast(convert_mask, tf.float32) * predict_h
        # convert_w = tf.cast(convert_mask, tf.float32) * predict_w
        #
        # predict_h = remain_h + convert_w
        # predict_w = remain_w + convert_h
        #
        # # [-270, -180)
        # cond4 = tf.cast(tf.logical_and(mask7, mask8), tf.float32) * 180.
        # predict_theta += cond4
        #
        # # [0, 90)
        # # cond2 = tf.cast(tf.logical_and(mask3, mask4), tf.float32) * 90.
        # # predict_theta -= cond2
        #
        # convert_mask1 = tf.logical_and(mask3, mask4)
        # remain_mask1 = tf.logical_not(convert_mask1)
        # predict_theta -= tf.cast(convert_mask1, tf.float32) * 90.
        #
        # remain_h = tf.cast(remain_mask1, tf.float32) * predict_h
        # remain_w = tf.cast(remain_mask1, tf.float32) * predict_w
        # convert_h = tf.cast(convert_mask1, tf.float32) * predict_h
        # convert_w = tf.cast(convert_mask1, tf.float32) * predict_w
        #
        # predict_h = remain_h + convert_w
        # predict_w = remain_w + convert_h
        #
        # # [90, 180)
        # cond3 = tf.cast(tf.logical_and(mask5, mask6), tf.float32) * 180.
        # predict_theta -= cond3

        decode_boxes = tf.transpose(tf.stack([predict_x_center, predict_y_center,
                                      predict_w, predict_h, predict_theta]))

        return decode_boxes


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None, name='encode'):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''

    with tf.variable_scope(name):
        x_center, y_center, w, h, theta = tf.unstack(unencode_boxes, axis=1)

        reference_x_center, reference_y_center, reference_w, reference_h, reference_theta = tf.unstack(reference_boxes, axis=1)

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8  # to avoid NaN in division and log below

        t_xcenter = (x_center - reference_x_center) / reference_w
        t_ycenter = (y_center - reference_y_center) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)
        t_theta = (theta - reference_theta) * pi / 180

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]
            t_theta *= scale_factors[4]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w, t_theta]))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    refer = tf.constant([[100, 100, 50, 100, -1]], tf.float32)
    encode = tf.constant([[0, 0, 0, 0, 2]], tf.float32)

    res = decode_boxes(encode, refer, [10., 10., 5., 5., 5.])

    with tf.Session() as sess:
        res1 = sess.run([res])
        print(res1)

