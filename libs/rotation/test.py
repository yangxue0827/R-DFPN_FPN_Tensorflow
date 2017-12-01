# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from libs.rotation import rotate_polygon_nms
import numpy as np

if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    # x_c, y_c, w, h = np.split(boxes, 5, axis=0)

    scores = np.array([0.99, 0.88, 0.66, 0.77])

    dets = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], np.transpose(scores)])

    dets = np.array(np.transpose(dets), np.float32)

    keep = rotate_polygon_nms.rotate_gpu_nms(dets, 0.7, device_id=1)

    print(keep)
