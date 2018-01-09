# -*- coding: utf-8 -*-
# @Time    : 2018/1/6 下午3:24
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import cv2
from kcftracker import KCFTracker
from hist import HistTracker
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
    def __init__(self, seqs_path, data_set_type, save_path=None):
        self.seq_loader = SequenceLoader(seqs_path, data_set_type)
        self.save_path = save_path

        if self.save_path and not os.path.exists(self.save_path):
            os.system('mkdir -p %s' % self.save_path)

        self.hist_tracker = None
        self.kcf_tracker = None

        self.last_frame = None
        self.seq_name = None
        self.action = None

        # hyper parameter
        self.precision_threshold = 40.0
        self.inteval = 1

    @property
    def gt_rects(self):
        return self.seq_loader.gt_rects

    # return frame0, frame1
    def reset(self, name):
        self.seq_loader.get_sequence(name)
        self.seq_name = name

        ret, img0 = self.seq_loader.get_cap().read()
        ret, img1 = self.seq_loader.get_cap().read()

        self.hist_tracker = HistTracker()
        self.kcf_tracker = KCFTracker(hog=False)

        self.hist_tracker.init(self.gt_rects[0], img0)
        self.kcf_tracker.init(self.gt_rects[0], img0)

        bg_img0 = self.get_sub_window(img0)
        bg_img1 = self.get_sub_window(img1)

        self.last_frame = img1
        self.frame_idx = 1

        cv2.destroyAllWindows()
        cv2.namedWindow(self.seq_name)
        cv2.namedWindow('%s color map' % self.seq_name)
        cv2.namedWindow('%s response map' % self.seq_name)

        cv2.moveWindow(self.seq_name, 20, 20)
        cv2.moveWindow('%s color map' % self.seq_name, 20 + self.hist_tracker.w + 100, 20)
        cv2.moveWindow('%s response map' % self.seq_name, 20 + self.hist_tracker.w + 100 + self.hist_tracker.bg_box[2] + 100, 20)

        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
        return bg_img0, bg_img1

    def get_sub_window(self, img):
        bg_image = self.hist_tracker.sub_window(img, self.hist_tracker.bg_box)
        return bg_image

    def show_tracking_result(self):
        predict_img = self.hist_tracker.get_predict_img(self.gt_rects[self.frame_idx - 1])
        cv2.putText(predict_img, 'S: %s' % self.action, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(self.seq_name, predict_img)
        if self.save_path:
            cv2.imwrite('%s/%d_predict.jpg'%(self.save_path, self.frame_idx -1), predict_img)

    def show_hist(self):
        color_map, response_map = self.hist_tracker.get_hist_map()
        cv2.imshow('%s color map' % self.seq_name, color_map)
        cv2.imshow('%s response map' % self.seq_name, response_map)

        if self.save_path:
            cv2.imwrite('%s/%d_color_map.jpg'%(self.save_path, self.frame_idx -1), color_map)
            cv2.imwrite('%s/%d_response.jpg'%(self.save_path, self.frame_idx -1), response_map)

    def show_all(self):
        self.show_tracking_result()
        self.show_hist()

        cv2.waitKey(self.inteval)

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

        self.action = action
        gt_box = self.gt_rects[self.frame_idx]
        gt_cx, gt_cy = gt_box[0] + gt_box[2] / 2.0, gt_box[1] + gt_box[3] / 2.0
        tg_cx, tg_cy = tg_box[0] + tg_box[2] / 2.0, tg_box[1] + tg_box[3] / 2.0

        # compute reward
        dis = math.sqrt((gt_cx - tg_cx) ** 2 + (gt_cy - tg_cy) ** 2)
        reward = 0.0 if dis > self.precision_threshold else 1 - (dis / self.precision_threshold)

        # crop img
        ret, self.last_frame = self.seq_loader.get_cap().read()
        self.frame_idx += 1
        bg_img = self.get_sub_window(self.last_frame) if ret else None

        return reward, bg_img, not ret
