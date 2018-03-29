# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.box_utils.coordinate_convert import forward_convert


def clip_boxes_to_img_boundaries(decode_boxes, img_shape):
    '''

    :param decode_boxes:
    :return: decode boxes, and already clip to boundaries
    '''

    with tf.name_scope('clip_boxes_to_img_boundaries'):

        xmin, ymin, xmax, ymax = tf.unstack(decode_boxes, axis=1)
        img_h, img_w = img_shape[1], img_shape[2]

        xmin = tf.maximum(xmin, 0.0)
        xmin = tf.minimum(xmin, tf.cast(img_w, tf.float32))

        ymin = tf.maximum(ymin, 0.0)
        ymin = tf.minimum(ymin, tf.cast(img_h, tf.float32))  # avoid xmin > img_w, ymin > img_h

        xmax = tf.minimum(xmax, tf.cast(img_w, tf.float32))
        ymax = tf.minimum(ymax, tf.cast(img_h, tf.float32))

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))


def filter_outside_boxes(boxes, img_w, img_h):
    '''
    :param anchors:boxes with format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param img_h: height of image
    :param img_w: width of image
    :return: indices of anchors that not outside the image boundary
    '''

    with tf.name_scope('filter_outside_boxes'):
        x1, y1, x2, y2, x3, y3, x4, y4 = tf.unstack(boxes, axis=1)

        x = tf.transpose(tf.stack([x1, x2, x3, x4]))
        y = tf.transpose(tf.stack([y1, y2, y3, y4]))
        xmin = tf.reduce_min(x, axis=1)
        ymin = tf.reduce_min(y, axis=1)
        xmax = tf.reduce_max(x, axis=1)
        ymax = tf.reduce_max(y, axis=1)

        xmin_index = tf.greater_equal(xmin, 0)
        ymin_index = tf.greater_equal(ymin, 0)
        xmax_index = tf.less_equal(xmax, tf.cast(img_w, tf.float32))
        ymax_index = tf.less_equal(ymax, tf.cast(img_h, tf.float32))

        indices = tf.transpose(tf.stack([xmin_index, ymin_index, xmax_index, ymax_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.equal(indices, 4))

        return tf.reshape(indices, [-1, ])


def nms_boxes(decode_boxes, scores, iou_threshold, max_output_size, name):
    '''
    1) NMS
    2) get maximum num of proposals
    :return: valid_indices
    '''

    valid_index = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        name=name
    )

    return valid_index


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with [0, 0, 1, 1, -90]
    :param boxes:
    :param scores: [-1, ]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    center_part = tf.zeros(shape=[pad_num, 2], dtype=boxes.dtype)
    size_part = tf.ones_like(center_part)
    theta_part = -90.0 * tf.ones(shape=[pad_num, 1])

    boxes_part = tf.concat([center_part, size_part, theta_part], axis=1)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, boxes_part], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores


def get_horizen_minAreaRectangle(boxs, with_label=True):

    rpn_proposals_boxes_convert = tf.py_func(forward_convert,
                                             inp=[boxs, with_label],
                                             Tout=tf.float32)

    if with_label:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 9])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])

        label = tf.unstack(rpn_proposals_boxes_convert, axis=1)[-1]

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)
        return tf.transpose(tf.stack([x_min, y_min, x_max, y_max, label], axis=0))
    else:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 8])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

    return tf.transpose(tf.stack([x_min, y_min, x_max, y_max], axis=0))