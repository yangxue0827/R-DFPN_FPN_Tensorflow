# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from libs.box_utils import coordinate_convert
from math import *


def image_crop_and_rotate(fp, boxes, stride, roi_size, ratio):
    """
    :param imgs: features map [h, w, c]
    :param boxes: boxes of batch image [x_c, y_c, w, h, theta]
    :return: rotate the feature map, and the boxes situation after rotating
    """

    boxes[2:3] //= stride

    theta = -1 * (boxes[:, -1] + 90)

    fp_rotates = []
    fp_h, fp_w = fp.shape[:2]
    rect_fp = ((0, 0), (fp_w, fp_h), -90)
    for i in range(boxes.shape[0]):
        rect = ((boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), boxes[i][4])

        int_pts = cv2.rotatedRectangleIntersection(rect, rect_fp)[1]

        if int_pts is not None:
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            order_pts = np.reshape(order_pts, [-1, 2])

            Xs = order_pts[:, 0]
            Ys = order_pts[:, 1]

            xmin = np.min(Xs)
            ymin = np.min(Ys)
            xmax = np.max(Xs)
            ymax = np.max(Ys)

            fp_crop = fp[int(max(ymin, 0)):int(min(ymax + 1, fp_h)), int(max(xmin, 0)):int(min(xmax + 1, fp_w)), :]

            if fp_crop.shape[0] == 0 or fp_crop.shape[1] == 0:
                crop_image = fp[0:1, 0:1]
            elif fp_crop.shape[0] <= 5 and fp_crop.shape[1] <= 5:
                crop_image = fp[0:1, 0:1]
            else:
                height, width = fp_crop.shape[:2]

                if boxes[i][2] < boxes[i][3]:
                    theta[i] -= 90

                height_new = int(width * fabs(sin(radians(theta[i]))) + height * fabs(cos(radians(theta[i]))))
                width_new = int(height * fabs(sin(radians(theta[i]))) + width * fabs(cos(radians(theta[i]))))

                mat_rotation = cv2.getRotationMatrix2D((width//2, height//2), theta[i], 1).astype(np.float32)

                mat_rotation[0, 2] += (width_new - width)//2
                mat_rotation[1, 2] += (height_new - height)//2

                fp_rotation = cv2.warpAffine(fp_crop, mat_rotation, (width_new, height_new), borderValue=(0, 0, 0))

                h, w = fp_rotation.shape[:2]
                crop_image = fp_rotation[max(0, (h - boxes[i][3])//2):int(min(h, (h + boxes[i][2])//2)),
                             max(0, (w - boxes[i][2]) // 2):int(min(w, (w + boxes[i][2])//2))]

        else:
            crop_image = fp[0:1, 0:1]

        s = int(roi_size / sqrt(ratio))
        l = int(roi_size * sqrt(ratio))

        fp_rotates.append(cv2.resize(crop_image, (l, s)))

    return np.array(fp_rotates, dtype=np.float32)


if __name__ == '__main__':
    coord = np.array([[130, 130, 130, 130, -45],
                      [1000, 1000, 100, 100, -90]])

    # img = np.zeros([500, 500, 4], dtype=np.uint8)
    img = cv2.imread('1.jpg')
    img = np.array(img, np.float32)

    # temp_fp, x_min, y_min, x_max, y_max = image_crop_and_rotate(img, coord, 1)
    # img_crop = temp_fp[0][y_min[0]:y_max[0], x_min[0]:x_max[0]]
    img_crop = image_crop_and_rotate(img, coord, 1, 50, 3)
    cv2.imwrite('test.jpg', img_crop[0])

