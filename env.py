# -*- coding: utf-8 -*-
# @Time    : 2018/1/6 下午3:24
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import cv2
from kcftracker import KCFTracker
from hist_v1 import HistTracker
import os
import math


class SequenceLoader:
    def __init__(self, seqs_path, data_set_type):
        self.seqs_path = seqs_path
        self.data_set_type = data_set_type
        self.gt_rects = []

        if self.data_set_type not in ['VOT', 'OTB']:
            print 'data set type is wrong!!!'
            exit()

    def read_VOT_gt(self, gt_path):
        self.gt_rects = []
        for line in open(gt_path, 'r'):
            x0, y0, x1, y1, x2, y2, x3, y3 = map(lambda x: float(x), line[:-1].split(','))
            max_x, min_x = max(x0, x1, x2, x3), min(x0, x1, x2, x3)
            max_y, min_y = max(y0, y1, y2, y3), min(y0, y1, y2, y3)
            x, y, w, h = min_x, min_y, (max_x - min_x), (max_y - min_y)
            self.gt_rects.append(map(int, (x, y, w, h)))

    def read_OTB_gt(self, gt_path):
        self.gt_rects = []
        for line in open(gt_path, 'r'):
            if line.find(',') != -1:
                x, y, w, h = map(lambda x: int(x), line[:-1].split(','))
            else:
                x, y, w, h = map(lambda x: int(x), line[:-1].split())
            self.gt_rects.append((x, y, w, h))

    def get_sequence(self, seq_name):
        seq_path = '%s/%s' % (self.seqs_path, seq_name)
        if self.data_set_type == 'VOT':
            self.cap = cv2.VideoCapture('%s/imgs/%%8d.jpg' % seq_path)
            self.read_VOT_gt('%s/groundtruth.txt' % seq_path)
        else:
            self.cap = cv2.VideoCapture('%s/img/%%4d.jpg' % seq_path)
            self.read_OTB_gt('%s/groundtruth_rect.txt' % seq_path)

    def get_cap(self):
        return self.cap

    def get_gt_rects(self):
        return self.gt_rects


class Env:
    def __init__(self, seqs_path, data_set_type):
        self.seq_loader = SequenceLoader(seqs_path, data_set_type)

        self.hist_tracker = None
        self.kcf_tracker = None

        self.last_frame = None

    @property
    def gt_rects(self):
        return self.seq_loader.gt_rects

    # return frame0, frame1
    def reset(self, name):
        self.seq_loader.get_sequence(name)
        ret, img0 = self.seq_loader.get_cap().read()
        ret, img1 = self.seq_loader.get_cap().read()

        self.hist_tracker = HistTracker()
        self.kcf_tracker = KCFTracker(hog=False)

        self.hist_tracker.init(self.gt_rects[0], img0)
        self.kcf_tracker.init(self.gt_rects[0], img0)

        self.last_frame = img1
        self.frame_idx = 1
        return img0, img1

    # return reward, next_frame, done
    def step(self, action):
        tg_box = (None, None, None, None)
        if action == 0:
            tg_box = self.hist_tracker.run(self.last_frame, None)
            self.kcf_tracker.run(self.last_frame, tg_box[:2])
        elif action == 1:
            tg_box = self.kcf_tracker.run(self.last_frame, None)
            tg_box = map(int, tg_box)
            self.hist_tracker.run(self.last_frame, tg_box[:2])
        else:
            print 'action is neither 0 nor 1!!!'
            exit()

        gt_box = self.gt_rects[self.frame_idx]
        gt_cx, gt_cy = gt_box[0] + gt_box[2] / 2.0, gt_box[1] + gt_box[3] / 2.0
        tg_cx, tg_cy = tg_box[0] + tg_box[2] / 2.0, tg_box[1] + tg_box[3] / 2.0

        # print 'gt:', gt_cx, gt_cy,
        # print 'pred:', tg_cx, tg_cy
        #
        # img = self.last_frame
        # gx, gy, gw, gh = gt_box
        # cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (255, 255, 0), 1)
        #
        # px, py, pw, ph = tg_box
        # cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 255), 1)
        #
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        reward = int(math.sqrt((gt_cx - tg_cx) ** 2 + (gt_cy - tg_cy) ** 2) <= 40)

        ret, self.last_frame = self.seq_loader.get_cap().read()
        self.frame_idx += 1
        return reward, self.last_frame, not ret
