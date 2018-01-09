# -*- coding: utf-8 -*-
# @Time    : 2018/1/5 上午9:44
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

"""
a naive version hist feature based Tracker
"""

import numpy as np
import cv2
import os


def normalize_255(a):
    max_val, min_val = a.max(), a.min()
    a = (a - min_val) / (max_val - min_val) * 255
    a = a.astype(np.uint8)

    return a


class HistTracker:
    def __init__(self):
        # frame size
        self.h = None
        self.w = None
        self.ext_len = None

        self.bg_pad = None
        self.fg_pad = None

        # will be changed before each update
        self.patch = None
        self.color_map = None
        self.response_map = None
        self.tg_box = None

        # will be changed after each update
        self.bg_box = None
        self.fg_box = None
        self.bg_hist = None
        self.fg_hist = None

        # hyper-parameter
        self.inner_padding_ratio = 0.6
        self.n_bins = 32
        self.w_bins = 256 / self.n_bins
        self.lr = 0

        # video configuration
        self.inteval = 1

    def init(self, gt_box, frame):
        self.h, self.w = frame.shape[:2]
        self.init_area_box(gt_box)
        self.init_ext_len()

        self.patch = self.sub_window(frame, self.bg_box)
        self.compute_hist()

    def compute_hist(self):
        def get_hist(patch, mask):
            bin_idx = (patch / self.w_bins).reshape(-1, 3)
            mask = mask.reshape(-1)

            hist = np.zeros((self.n_bins, self.n_bins, self.n_bins))
            for i, (r, g, b) in enumerate(bin_idx):
                hist[r, g, b] += mask[i]

            return hist / mask.sum()

        # create bg_mask
        bg_bound = self.bg_pad
        bg_mask = np.ones((self.bg_box[3], self.bg_box[2]))
        bg_mask[bg_bound:-bg_bound, bg_bound:-bg_bound] = 0

        # create fg_mask
        fg_bound = self.bg_pad + self.fg_pad
        fg_mask = np.zeros((self.bg_box[3], self.bg_box[2]))
        fg_mask[fg_bound:-fg_bound, fg_bound:-fg_bound] = 1

        if self.bg_hist is None:
            self.bg_hist = get_hist(self.patch, bg_mask)
            self.fg_hist = get_hist(self.patch, fg_mask)
        else:
            self.bg_hist = (1 - self.lr) * self.bg_hist + self.lr * get_hist(self.patch, bg_mask)
            self.fg_hist = (1 - self.lr) * self.fg_hist + self.lr * get_hist(self.patch, fg_mask)

    def compute_color_map(self):
        def get_color_map(bin_idx, hist):
            color_map = hist[bin_idx[:, 0], bin_idx[:, 1], bin_idx[:, 2]]
            return color_map.reshape(self.bg_box[3], self.bg_box[2])

        bin_idx = (self.patch / self.w_bins).reshape(-1, 3)
        bg_color_map = get_color_map(bin_idx, self.bg_hist)
        fg_color_map = get_color_map(bin_idx, self.fg_hist)

        self.color_map = fg_color_map / (fg_color_map + bg_color_map)
        self.color_map[np.isnan(self.color_map)] = 0
        # self.color_map = normalize_255(self.color_map)

    def compute_response_map(self):
        top, bottom, left, right = self.ext_len
        patch = np.lib.pad(self.color_map, ((top, bottom - 1), (left, right - 1)), 'constant', constant_values=0)
        bg_w, bg_h = self.bg_box[2], self.bg_box[3]

        # compute response map
        # patch = patch.astype(np.float64) / 255

        SAT = cv2.integral(patch)
        self.response_map = SAT[:bg_h, :bg_w] + SAT[-bg_h:, -bg_w:] - SAT[:bg_h, -bg_w:] - SAT[-bg_h:, :bg_w]
        # self.response_map = self.response_map / (self.tg_box[2] * self.tg_box[3])
        # self.response_map = normalize_255(self.response_map)

        # h,w = self.response_map.shape
        # for i in xrange(h):
        #     for j in xrange(w):
        #         print self.response_map[i,j],
        #     print
        # exit()

        # compute pred_cpos
        idxs = np.where(self.response_map == self.response_map.max())
        # print  idxs[1][0], idxs[0][0]

        pred_cpos = (self.bg_box[0] + idxs[1][0], self.bg_box[1] + idxs[0][0])

        return pred_cpos

    def sub_window(self, img, sub_box):
        sub_lt_x, sub_lt_y, sub_w, sub_h = sub_box
        h_idx = np.arange(sub_lt_y, sub_lt_y + sub_h)
        w_idx = np.arange(sub_lt_x, sub_lt_x + sub_w)

        h_idx[h_idx < 0] = 0
        h_idx[h_idx >= self.h] = self.h - 1
        w_idx[w_idx < 0] = 0
        w_idx[w_idx >= self.w] = self.w - 1

        # TODO: 
        # img[[0, self.h - 1], :, :] = 0
        # img[:, [0, self.w - 1], :] = 0
        ret = img[h_idx, :, :][:, w_idx, :]

        return ret

    def init_area_box(self, gt_box):
        gt_x, gt_y, gt_w, gt_h = gt_box

        self.tg_box = (gt_x, gt_y, gt_w, gt_h)

        # average padding width
        self.bg_pad = int((gt_w + gt_h) / 8.0)
        self.bg_box = (gt_x - self.bg_pad, gt_y - self.bg_pad, gt_w + 2 * self.bg_pad, gt_h + 2 * self.bg_pad)

        self.fg_pad = int(self.inner_padding_ratio * self.bg_pad)
        self.fg_box = (gt_x + self.fg_pad, gt_y + self.fg_pad, gt_w - 2 * self.fg_pad, gt_h - 2 * self.fg_pad)

        self.old_bg_box = self.bg_box
        self.old_fg_box = self.fg_box

    def init_ext_len(self):
        tg_w, tg_h = self.tg_box[2:]
        top = tg_h / 2
        bottom = tg_h - top
        left = tg_w / 2
        right = tg_w - left

        self.ext_len = (top, bottom, left, right)

    def run(self, img, lt_pos):
        if lt_pos == None:
            self.patch = self.sub_window(img, self.bg_box)
            self.compute_color_map()
            pred_cpos = self.compute_response_map()
            self.pred_cpos = pred_cpos
        else:
            pred_cpos = (lt_pos[0] + self.ext_len[2], lt_pos[1] + self.ext_len[0])
            self.pred_cpos = pred_cpos

        # update tg_box, then update bg_box, fg_box, bg_hist, fg_hist
        self.update(pred_cpos)
        self.patch = self.sub_window(img, self.tg_box)
        self.compute_hist()

        return self.tg_box

    def update(self, pred_cpos):
        pred_x, pred_y = pred_cpos[0] - self.ext_len[2], pred_cpos[1] - self.ext_len[0]
        # print 'pred:', pred_x, pred_y
        self.tg_box = (pred_x, pred_y, self.tg_box[2], self.tg_box[3])

        self.old_bg_box = self.bg_box
        self.old_fg_box = self.fg_box

        # update bg_box, fg_box
        self.bg_box = (pred_x - self.bg_pad, pred_y - self.bg_pad, self.bg_box[2], self.bg_box[3])
        self.fg_box = (pred_x + self.fg_pad, pred_y + self.fg_pad, self.fg_box[2], self.fg_box[3])
