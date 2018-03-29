# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def forward_convert(coordinate, with_label=False, mode=None):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if mode:
        coordinate = coordinate_present_convert(coordinate, mode)

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            boxes.append(np.reshape(box, [-1, ]))

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinate, with_label=True, mode=None):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :param mode: -1 convert coords range to [-90, 90), 1 convert coords range to [-90, 0), default(1)
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])
        if mode:
            boxes = np.array(boxes, dtype=np.float32)
            boxes_temp = coordinate_present_convert(boxes[:, :-1], mode)
            boxes[:, :-1] = boxes_temp

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])
        if mode:
            boxes = coordinate_present_convert(np.array(boxes, dtype=np.float32), mode)

    return np.array(boxes, dtype=np.float32)


def coordinate_present_convert(coords, mode=1):
    """
    :param coords: shape [-1, 5]
    :param mode: -1 convert coords range to [-90, 90), 1 convert coords range to [-90, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [-90, 90)
    if mode == -1:
        w, h = coords[:, 2], coords[:, 3]

        remain_mask = np.greater(w, h)
        convert_mask = np.logical_not(remain_mask).astype(np.int32)
        remain_mask = remain_mask.astype(np.int32)

        remain_coords = coords * np.reshape(remain_mask, [-1, 1])

        coords[:, [2, 3]] = coords[:, [3, 2]]
        coords[:, -1] += 90

        convert_coords = coords * np.reshape(convert_mask, [-1, 1])

        coords_new = remain_coords + convert_coords

    # angle range from [-90, 90) to [-90, 0)
    elif mode == 1:
        theta = coords[:, -1]
        remain_mask = np.logical_and(np.greater_equal(theta, -90), np.less(theta, 0))
        convert_mask = np.logical_not(remain_mask)

        remain_coords = coords * np.reshape(remain_mask, [-1, 1])

        coords[:, [2, 3]] = coords[:, [3, 2]]
        coords[:, -1] -= 90

        convert_coords = coords * np.reshape(convert_mask, [-1, 1])

        coords_new = remain_coords + convert_coords
    else:
        raise Exception('mode error!')

    return np.array(coords_new, dtype=np.float32)


if __name__ == '__main__':
    coord = np.array([[150, 150, 50, 100, -90],
                      [150, 150, 100, 50, -90],
                      [150, 150, 50, 100, -45],
                      [150, 150, 100, 50, -45]])

    coord1 = np.array([[150, 150, 100, 50, 0, 1],
                      [150, 150, 100, 50, -90, 1],
                      [150, 150, 100, 50, 45, 1],
                      [150, 150, 100, 50, -45, 1]])

    coord2 = forward_convert(coord1, True)
    # coord3 = forward_convert(coord1, mode=-1)
    print(coord2)
    # print(coord3-coord2)
    # coord_label = np.array([[167., 203., 96., 132., 132., 96., 203., 167., 1.]])
    #
    # coord4 = back_forward_convert(coord_label, mode=1)
    # coord5 = back_forward_convert(coord_label)

    # print(coord4)
    # print(coord5)

    # coord3 = coordinate_present_convert(coord, -1)
    # print(coord3)
    # coord4 = coordinate_present_convert(coord3, mode=1)
    # print(coord4)

