# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys

sys.path.append('../')
import tensorflow as tf
import numpy as np
import os
import time

from data.io.read_tfrecord import next_batch
from libs.configs import cfgs
from libs.networks.network_factory import get_flags_byname
from libs.networks.network_factory import get_network_byname
from help_utils.tools import *
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn
from libs.box_utils.coordinate_convert import back_forward_convert
import cv2
from help_utils import help_utils
from tools import restore_model
from libs.box_utils import nms_rotate


RESTORE_FROM_RPN = False
FLAGS = get_flags_byname(cfgs.NET_NAME)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test(img_num):
    with tf.Graph().as_default():

        # img = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.SHORT_SIDE_LEN,
                       is_training=False)

        gtboxes_and_label = tf.py_func(back_forward_convert,
                                       inp=[tf.squeeze(gtboxes_and_label_batch, 0)],
                                       Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)
        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=gtboxes_and_label,
                            is_training=False,
                            share_head=False,
                            share_net=share_net,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            anchor_angles=cfgs.ANCHOR_ANGLES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            anchor_stride=cfgs.ANCHOR_STRIDE,
                            pool_stride=cfgs.POOL_STRIDE,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            kernel_size=cfgs.KERNEL_SIZE,
                            use_angles_condition=True,
                            anchor_angle_threshold=cfgs.RPN_ANCHOR_ANGLES_THRESHOLD,
                            nms_angle_threshold=cfgs.RPN_NMS_ANGLES_THRESHOLD,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                            scope='')

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]
        _, _, rpn_top_k_boxes, rpn_top_k_scores = rpn.rpn_losses()

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                             feature_pyramid=rpn.feature_pyramid,
                                             rpn_proposals_boxes=rpn_proposals_boxes,
                                             rpn_proposals_scores=rpn_proposals_scores,
                                             stop_gradient_for_proposals=False,
                                             img_shape=tf.shape(img_batch),
                                             roi_size=cfgs.ROI_SIZE,
                                             roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                             scale_factors=cfgs.SCALE_FACTORS,
                                             gtboxes_and_label=None,
                                             fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             top_k_nms=cfgs.FAST_RCNN_TOP_K_NMS,
                                             nms_angle_threshold=cfgs.FAST_RCNN_NMS_ANGLES_THRESHOLD,
                                             use_angle_condition=False,
                                             level=cfgs.LEVEL,
                                             fast_rcnn_maximum_boxes_per_img=100,
                                             fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                             # show detections which score >= 0.6
                                             num_classes=cfgs.CLASS_NUM,
                                             fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                             fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                             fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                             boxes_angle_threshold=cfgs.FAST_RCNN_BOXES_ANGLES_THRESHOLD,
                                             use_dropout=cfgs.USE_DROPOUT,
                                             weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                             is_training=False)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
            fast_rcnn.fast_rcnn_predict()

        ##############################################################################################
        if cfgs.NEED_AUXILIARY:
            predict_boxes = tf.concat([fast_rcnn_decode_boxes, rpn_top_k_boxes], axis=0)
            predict_scores = tf.concat([fast_rcnn_score, rpn_top_k_scores - 0.2], axis=0)
            rpn_top_k_label = tf.ones([tf.shape(rpn_top_k_scores)[0], ], tf.int64)
            labels = tf.concat([detection_category, rpn_top_k_label], axis=0)

            # valid_indices = nms_rotate.nms_rotate(decode_boxes=predict_boxes,
            #                                       scores=predict_scores,
            #                                       iou_threshold=0.15,
            #                                       max_output_size=30,
            #                                       use_angle_condition=False,
            #                                       angle_threshold=15,
            #                                       use_gpu=True)
            valid_indices = tf.py_func(nms_rotate.nms_rotate_cpu,
                                       inp=[predict_boxes, predict_scores,
                                            tf.constant(0.15, tf.float32), tf.constant(30, tf.float32)],
                                       Tout=tf.int64)

            fast_rcnn_decode_boxes = tf.gather(predict_boxes, valid_indices)
            fast_rcnn_score = tf.gather(predict_scores, valid_indices)
            detection_category = tf.gather(labels, valid_indices)

            ##############################################################################################

        # train
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            for i in range(img_num):

                start = time.time()

                _img_name_batch, _img_batch, _gtboxes_and_label, _fast_rcnn_decode_boxes, \
                _fast_rcnn_score, _detection_category \
                    = sess.run([img_name_batch, img_batch, gtboxes_and_label, fast_rcnn_decode_boxes,
                                fast_rcnn_score, detection_category])
                end = time.time()

                _img_batch = np.squeeze(_img_batch, axis=0)

                _img_batch_fpn = help_utils.draw_box_cv(_img_batch,
                                                        boxes=_fast_rcnn_decode_boxes,
                                                        labels=_detection_category,
                                                        scores=_fast_rcnn_score)
                mkdir(cfgs.TEST_SAVE_PATH)
                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_fpn.jpg'.format(str(_img_name_batch[0]).split('.tif')[0]),
                            _img_batch_fpn)

                # _gtboxes_and_label_batch = np.squeeze(_gtboxes_and_label_batch, axis=0)

                temp_label = np.reshape(_gtboxes_and_label[:, -1:], [-1, ]).astype(np.int64)
                _img_batch_gt = help_utils.draw_box_cv(_img_batch,
                                                       boxes=_gtboxes_and_label[:, :-1],
                                                       labels=temp_label,
                                                       scores=None)

                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/{}_gt.jpg'.format(str(_img_name_batch[0]).split('.tif')[0]),
                            _img_batch_gt)

                view_bar('{} image cost {}s'.format(str(_img_name_batch[0]), (end - start)), i+1, img_num)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    img_num = 200
    test(img_num)










