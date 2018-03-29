# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def sample_balance(source, indices, pad_num):
    indices_tile = tf.tile(indices, [pad_num, ])

    indices_tile = tf.random_shuffle(indices_tile)
    indices_pad = tf.slice(indices_tile, begin=[0], size=[pad_num])

    boxes_pad = tf.gather(source, indices_pad)

    ws = boxes_pad[:, 2]
    hs = boxes_pad[:, 3]
    thetas = boxes_pad[:, 4]

    hs_offset = (tf.truncated_normal(shape=tf.shape(indices_pad)) - 0.5) * 0.1 * hs
    ws_offset = (tf.truncated_normal(shape=tf.shape(indices_pad)) - 0.5) * 0.1 * ws
    thetas_offset = (tf.truncated_normal(shape=tf.shape(indices_pad)) - 0.5) * 0.1 * thetas

    hs += hs_offset
    ws += ws_offset
    thetas += thetas_offset

    boxes_new = tf.transpose(tf.stack([boxes_pad[:, 0], boxes_pad[:, 1], ws, hs, thetas]))

    return boxes_new, indices_pad
