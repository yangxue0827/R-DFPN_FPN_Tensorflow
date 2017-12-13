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
import pickle
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn
from libs.box_utils.coordinate_convert import back_forward_convert
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle
from libs.label_name_dict.label_dict import *
from help_utils.tools import view_bar
from tools import restore_model
from libs.box_utils import iou_rotate
from libs.box_utils import nms_rotate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_dict_packle(_gtboxes_and_label, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category):

    gtbox_list = []
    predict_list = []

    for j, box in enumerate(_gtboxes_and_label):
        bbox_dict = {}
        bbox_dict['bbox'] = np.array(_gtboxes_and_label[j, :-1], np.float64)
        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label[j, -1])]
        gtbox_list.append(bbox_dict)

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue
        else:
            temp_dict = {}
            temp_dict['name'] = label

            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
            temp_boxes = _fast_rcnn_decode_boxes[ind]
            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
            temp_dict['bbox'] = np.array(np.concatenate([temp_boxes, temp_score], axis=1), np.float64)
            predict_list.append(temp_dict)
    return gtbox_list, predict_list


def eval_dict_convert(img_num, mode):
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

        gtboxes_and_label_minAreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)

        gtboxes_and_label_minAreaRectangle = tf.reshape(gtboxes_and_label_minAreaRectangle, [-1, 5])

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
                            share_head=cfgs.SHARE_HEAD,
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
                            use_angles_condition=False,
                            anchor_angle_threshold=cfgs.RPN_ANCHOR_ANGLES_THRESHOLD,
                            nms_angle_threshold=cfgs.RPN_NMS_ANGLES_THRESHOLD,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            # iou>=0.7 is positive box, iou< 0.3 is negative
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=cfgs.IS_FILTER_OUTSIDE_BOXES,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                            scope='')

        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]
        _, _, rpn_predict_boxes, rpn_predict_scores = rpn.rpn_losses()

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
            predict_boxes = tf.concat([fast_rcnn_decode_boxes, rpn_predict_boxes], axis=0)
            predict_scores = tf.concat([fast_rcnn_score, rpn_predict_scores - 0.2], axis=0)
            rpn_predict_label = tf.ones([tf.shape(rpn_predict_scores)[0], ], tf.int64)
            labels = tf.concat([detection_category, rpn_predict_label], axis=0)

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
        if mode == 0:
            fast_rcnn_decode_boxes = get_horizen_minAreaRectangle(fast_rcnn_decode_boxes, False)

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

            gtboxes_dict = {}
            predict_dict = {}

            for i in range(img_num):

                start = time.time()

                _img_name_batch, _img_batch, _gtboxes_and_label, _fast_rcnn_decode_boxes, \
                _gtboxes_and_label_minAreaRectangle, _fast_rcnn_score, _detection_category \
                    = sess.run([img_name_batch, img_batch, gtboxes_and_label, fast_rcnn_decode_boxes,
                                gtboxes_and_label_minAreaRectangle, fast_rcnn_score, detection_category])
                end = time.time()

                # gtboxes convert dict
                gtboxes_dict[str(_img_name_batch[0])] = []
                predict_dict[str(_img_name_batch[0])] = []

                # for j, box in enumerate(_gtboxes_and_label):
                #     bbox_dict = {}
                #     bbox_dict['bbox'] = np.array(_gtboxes_and_label[j, :-1], np.float64)
                #     bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label[j, -1])]
                #     gtbox_dict[str(_img_name_batch[0])].append(bbox_dict)
                #
                # for label in NAME_LABEL_MAP.keys():
                #     if label == 'back_ground':
                #         continue
                #     else:
                #         temp_dict = {}
                #         temp_dict['name'] = label
                #
                #         ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
                #         temp_boxes = _fast_rcnn_decode_boxes[ind]
                #         temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
                #         temp_dict['bbox'] = np.array(np.concatenate([temp_boxes, temp_score], axis=1), np.float64)
                #         predict_dict[str(_img_name_batch[0])].append(temp_dict)

                if mode == 0:
                    gtboxes_list, predict_list = \
                        make_dict_packle(_gtboxes_and_label_minAreaRectangle, _fast_rcnn_decode_boxes,
                                         _fast_rcnn_score, _detection_category)
                else:
                    gtboxes_list, predict_list = \
                        make_dict_packle(_gtboxes_and_label, _fast_rcnn_decode_boxes,
                                         _fast_rcnn_score, _detection_category)

                gtboxes_dict[str(_img_name_batch[0])].extend(gtboxes_list)
                predict_dict[str(_img_name_batch[0])].extend(predict_list)

                view_bar('{} image cost {}s'.format(str(_img_name_batch[0]), (end - start)), i + 1, img_num)

            fw1 = open('gtboxes_dict.pkl', 'w')
            fw2 = open('predict_dict.pkl', 'w')
            pickle.dump(gtboxes_dict, fw1)
            pickle.dump(predict_dict, fw2)
            fw1.close()
            fw2.close()
            coord.request_stop()
            coord.join(threads)


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_single_label_dict(predict_dict, gtboxes_dict, label):
    rboxes = {}
    gboxes = {}
    rbox_images = predict_dict.keys()
    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        for pre_box in predict_dict[rbox_image]:
            if pre_box['name'] == label and len(pre_box['bbox']) != 0:
                rboxes[rbox_image] = [pre_box]

                gboxes[rbox_image] = []

                for gt_box in gtboxes_dict[rbox_image]:
                    if gt_box['name'] == label:
                        gboxes[rbox_image].append(gt_box)
    return rboxes, gboxes


def voc_eval(rboxes, gboxes, iou_th, use_07_metric, mode):
    rbox_images = rboxes.keys()
    fp = np.zeros(len(rbox_images))
    tp = np.zeros(len(rbox_images))
    box_num = 0

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        if len(rboxes[rbox_image][0]['bbox']) > 0:

            rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
            if len(gboxes[rbox_image]) > 0:
                gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
                box_num = box_num + len(gbox_list)
                gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
                confidence = rbox_lists[:, -1]
                box_index = np.argsort(-confidence)

                rbox_lists = rbox_lists[box_index, :]
                for rbox_list in rbox_lists:
                    if mode == 0:
                        ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
                        iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
                        ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
                        iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih  # 重合的面积

                        # union
                        uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                               (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                               (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)  # 并的面积
                        overlaps = inters / uni
                    else:
                        overlaps = iou_rotate.iou_rotate_calculate1(np.array([rbox_list[:-1]]),
                                                                    gbox_list,
                                                                    use_gpu=False)[0]

                    ovmax = np.max(overlaps)  # 取最大重合的面积
                    # print(ovmax)
                    jmax = np.argmax(overlaps)  # 取最大重合面积的索引
                    if ovmax > iou_th:
                        if gbox_list[jmax, -1] == 0:
                            tp[i] += 1
                            gbox_list[jmax, -1] = 1
                        else:
                            fp[i] += 1
                    else:
                        fp[i] += 1

            else:
                fp[i] += len(rboxes[rbox_image][0]['bbox'])
        else:
            continue
    rec = np.zeros(len(rbox_images))
    prec = np.zeros(len(rbox_images))
    if box_num == 0:
        for i in range(len(fp)):
            if fp[i] != 0:
                prec[i] = 0
            else:
                prec[i] = 1
    else:

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / (box_num + cfgs.EPSILON)
    # avoid division by zero in case first detection matches a difficult ground ruth
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


if __name__ == '__main__':
    img_num = 1706
    # 0: horizontal standard 1: rotate standard
    mode = 1
    eval_dict_convert(img_num, mode)

    fr1 = open('predict_dict.pkl', 'r')
    fr2 = open('gtboxes_dict.pkl', 'r')
    predict_dict = pickle.load(fr1)
    gtboxes_dict = pickle.load(fr2)

    R, P, mAP, F = 0, 0, 0, 0

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)

        rec, prec, ap = voc_eval(rboxes, gboxes, 0.5, False, mode=mode)

        recall = rec[-1]
        precision = prec[-1]
        F_measure = (2 * precision * recall) / (recall + precision)
        print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format(label, recall, precision, ap, F_measure))
        R += recall
        P += precision
        mAP += ap
        F += F_measure
    print('\n{}\tR:{}\tP:{}\tap:{}\tF:{}'.format('Final', R / cfgs.CLASS_NUM,
                                                 P / cfgs.CLASS_NUM,
                                                 mAP / cfgs.CLASS_NUM,
                                                 F / cfgs.CLASS_NUM))

    fr1.close()
    fr2.close()









