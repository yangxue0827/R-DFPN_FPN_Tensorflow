# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from math import pi
import os
from libs.box_utils.show_box_in_tensor import draw_boxes_with_categories, draw_box_with_color
import cv2
import numpy as np


def enum_scales(base_anchor, anchor_scales, name='enum_scales'):
    """
    :param base_anchor: [x_center, y_center, w, h]
    :param anchor_scales: different scales, like [0.5, 1., 2.]
    :param name: function name
    :return: base anchors in different scales.
            Example: [[0, 0, 128, 128], [0, 0, 256, 256], [0, 0, 512, 512]]
    """

    with tf.variable_scope(name):
        anchor_scales = tf.reshape(anchor_scales, [-1, 1])

        return base_anchor * anchor_scales


def enum_ratios(anchors, anchor_ratios, name='enum_ratios'):

    """
    :param anchors: base anchors in different scales
    :param anchor_ratios: ratios = h / w
    :param name: function name
    :return: base anchors in different scales and ratios
    """
    with tf.variable_scope(name):
        # for base anchor, h == w
        _, _, hs, ws = tf.unstack(anchors, axis=1)
        sqrt_ratios = tf.sqrt(anchor_ratios)
        sqrt_ratios = tf.expand_dims(sqrt_ratios, axis=1)
        ws = tf.reshape(ws / sqrt_ratios, [-1])
        hs = tf.reshape(hs * sqrt_ratios, [-1])

        num_anchors_per_location = tf.shape(ws)[0]

        return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location, ]),
                                      tf.zeros([num_anchors_per_location, ]),
                                      ws, hs]))


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles, featuremaps_height,
                 featuremaps_width, stride, name='make_ratate_anchors'):

    """
    :param base_anchor_size: base anchor size in different scales
    :param anchor_scales: anchor scales
    :param anchor_ratios: anchor ratios
    :param anchor_angles: anchor angles
    :param featuremaps_height: height of featuremaps
    :param featuremaps_width: width of featuremaps
    :param stride: distance of anchor centers (h_original / h_featuremap or w_original / w_featuremap)
    :param name: function name
    :return: anchors of shape [w * h * len(anchor_scales) * len(anchor_ratios), 5]
    """

    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)
        base_anchors = enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)
        anchor_angles = tf.expand_dims(anchor_angles, axis=1)
        base_anchors = tf.concat([base_anchors, anchor_angles], axis=1)

        x_c, y_c, hs, ws, angles = tf.unstack(base_anchors, axis=1)

        x_centers = tf.range(tf.cast(featuremaps_width, tf.float32), dtype=tf.float32) * stride
        y_centers = tf.range(tf.cast(featuremaps_height, tf.float32), dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        box_centers = tf.stack([x_centers, y_centers], axis=2)
        box_centers = tf.reshape(box_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        angles = tf.tile(angles, [tf.cast(tf.shape(box_sizes)[0] / tf.shape(angles)[0], tf.int32)])
        angles = tf.expand_dims(angles, axis=1)
        final_anchors = tf.concat([box_centers, box_sizes, angles], axis=1)

        return final_anchors


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    base_anchor = tf.constant([256], dtype=tf.float32)
    anchor_scales = tf.constant([1.0], dtype=tf.float32)
    anchor_ratios = tf.constant([1 / 3., 3., 1 / 3., 3., 1 / 3., 3.], dtype=tf.float32)
    anchor_angles = tf.constant([-90, -90, -30., -30., -60., -60.], dtype=tf.float32)
    temp = enum_scales(base_anchor, anchor_scales)

    anchors = make_anchors(64, anchor_scales, anchor_ratios, anchor_angles,
                           featuremaps_height=38,
                           featuremaps_width=50,
                           stride=16)

    img = tf.ones([38*16, 50*16, 3])
    img = tf.expand_dims(img, axis=0)

    img1 = draw_box_with_color(img, anchors[5502:5508], text=tf.shape(anchors)[1])

    with tf.Session() as sess:
        temp1, _img1 = sess.run([anchors, img1])

        _img1 = _img1[0]

        cv2.imwrite('rotate_anchors.jpg', _img1)
        cv2.waitKey(0)

        print(temp1)
        print('debug')

