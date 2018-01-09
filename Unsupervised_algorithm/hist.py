# -*- coding: utf-8 -*-
# @Time    : 2018/1/5 上午9:44
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn
"""
version 1.3.1
resize bg_box to standard area(e.g, 150 * 150) to speed up compute hist
change run: it will predict pos every frame, even if no useful
TODO: still too slow
"""
import numpy as np
import cv2
import math
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
        self.hist_scale_rate = None

        # will be changed before each update
        self.img = None
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
        self.inner_padding_ratio = 0.2
        self.n_bins = 32
        self.w_bins = 256 / self.n_bins
        self.lr = 0.04
        self.standard_area = 150.0 ** 2

        # video configuration
        self.inteval = 1

    def init(self, gt_box, frame):
        self.h, self.w = frame.shape[:2]
        self.init_area_box(gt_box)
        self.init_ext_len()

        self.img = frame
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

        scaled_bg_w, scaled_bg_h = int(self.bg_box[2] * self.hist_scale_rate), int(self.bg_box[3] * self.hist_scale_rate)

        # create bg_mask
        bg_bound = int(self.bg_pad * self.hist_scale_rate)
        bg_mask = np.ones((scaled_bg_h, scaled_bg_w))
        bg_mask[bg_bound:-bg_bound, bg_bound:-bg_bound] = 0

        # create fg_mask
        fg_bound = int((self.bg_pad + self.fg_pad) * self.hist_scale_rate)
        fg_mask = np.zeros((scaled_bg_h, scaled_bg_w))
        fg_mask[fg_bound:-fg_bound, fg_bound:-fg_bound] = 1

        if self.hist_scale_rate != 1:
            scaled_patch = cv2.resize(self.patch, (scaled_bg_w, scaled_bg_h))
        else:
            scaled_patch = self.patch

        if self.bg_hist is None:
            self.bg_hist = get_hist(scaled_patch, bg_mask)
            self.fg_hist = get_hist(scaled_patch, fg_mask)
        else:
            self.bg_hist = (1 - self.lr) * self.bg_hist + self.lr * get_hist(scaled_patch, bg_mask)
            self.fg_hist = (1 - self.lr) * self.fg_hist + self.lr * get_hist(scaled_patch, fg_mask)

    def clean_color_map(self):
        h, w = self.color_map.shape
        max_v = self.color_map.max()
        # self.color_map = normalize_255(self.color_map)
        self.color_map[self.color_map <= max_v / 3.0] = 0
        self.color_map[self.color_map > max_v / 3.0] = self.color_map[self.color_map > max_v / 3.0] / max_v * 255

        self.color_map = self.color_map.astype(np.uint8)

        self.color_map = cv2.medianBlur(self.color_map, 5)

        # self.color_map = cv2.medianBlur(self.color_map, 7) / 255.0
        # self.color_map = self.color_map * max_v

    def compute_color_map(self):
        def get_color_map(bin_idx, hist):
            color_map = hist[bin_idx[:, 0], bin_idx[:, 1], bin_idx[:, 2]]
            return color_map.reshape(self.bg_box[3], self.bg_box[2])

        bin_idx = (self.patch / self.w_bins).reshape(-1, 3)
        bg_color_map = get_color_map(bin_idx, self.bg_hist)
        fg_color_map = get_color_map(bin_idx, self.fg_hist)

        self.color_map = fg_color_map / (fg_color_map + bg_color_map)
        self.color_map[np.isnan(self.color_map)] = 0

        self.clean_color_map()
        # self.color_map = normalize_255(self.color_map)

    def compute_response_map(self):
        odd = lambda x: x if x % 2 == 1 else x + 1
        w, h = self.tg_box[2:]
        w, h = odd(w), odd(h)

        self.response_map = cv2.GaussianBlur(self.color_map, (w, h), 0, borderType=cv2.BORDER_CONSTANT)

        # compute pred_cpos
        idxs = np.where(self.response_map == self.response_map.max())
        x, y = int(idxs[1].mean()), int(idxs[0].mean())

        pred_cpos = (self.bg_box[0] + x, self.bg_box[1] + y)

        return pred_cpos

    def sub_window(self, img, sub_box):
        sub_lt_x, sub_lt_y, sub_w, sub_h = sub_box
        h_idx = np.arange(sub_lt_y, sub_lt_y + sub_h)
        w_idx = np.arange(sub_lt_x, sub_lt_x + sub_w)

        h_idx[h_idx < 0] = 0
        h_idx[h_idx >= self.h] = self.h - 1
        w_idx[w_idx < 0] = 0
        w_idx[w_idx >= self.w] = self.w - 1

        ret = img[h_idx, :, :][:, w_idx, :]
        return ret

    def init_area_box(self, gt_box):
        gt_x, gt_y, gt_w, gt_h = gt_box

        self.tg_box = (gt_x, gt_y, gt_w, gt_h)

        # average padding width
        self.bg_pad = int((gt_w + gt_h) / 4.0)
        self.bg_box = (gt_x - self.bg_pad, gt_y - self.bg_pad, gt_w + 2 * self.bg_pad, gt_h + 2 * self.bg_pad)

        self.fg_pad = int(self.inner_padding_ratio * self.bg_pad)
        self.fg_box = (gt_x + self.fg_pad, gt_y + self.fg_pad, gt_w - 2 * self.fg_pad, gt_h - 2 * self.fg_pad)

        if self.bg_box[2] * self.bg_box[3] > self.standard_area:
            self.hist_scale_rate = math.sqrt(self.standard_area / (self.bg_box[2] * self.bg_box[3]))

        else:
            self.hist_scale_rate = 1

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
        self.patch = self.sub_window(img, self.bg_box)
        self.compute_color_map()
        pred_cpos = self.compute_response_map()

        if lt_pos != None:
            pred_cpos = (lt_pos[0] + self.ext_len[2], lt_pos[1] + self.ext_len[0])
        self.pred_cpos = pred_cpos

        # update tg_box, then update bg_box, fg_box, bg_hist, fg_hist
        self.img = img
        self.update(pred_cpos)
        self.patch = self.sub_window(img, self.bg_box)
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

    def get_predict_img(self, gt_box):
        # TODO: here changing img maybe influnce other algorithm
        px, py, pw, ph = self.tg_box
        cv2.rectangle(self.img, (px, py), (px + pw, py + ph), (0, 255, 255), 1)

        bg_x, bg_y, bg_w, bg_h = self.old_bg_box
        cv2.rectangle(self.img, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (255, 0, 255), 1)

        fg_x, fg_y, fg_w, fg_h = self.old_fg_box
        cv2.rectangle(self.img, (fg_x, fg_y), (fg_x + fg_w, fg_y + fg_h), (255, 0, 255), 1)

        gx, gy, gw, gh = gt_box
        cv2.rectangle(self.img, (gx, gy), (gx + gw, gy + gh), (255, 255, 0), 1)

        return self.img

    def get_hist_map(self):
        rgb_response_map = cv2.cvtColor(normalize_255(self.response_map), cv2.COLOR_GRAY2RGB)
        rgb_color_map = cv2.cvtColor(normalize_255(self.color_map), cv2.COLOR_GRAY2RGB)

        bg_pred_x, bg_pred_y = self.pred_cpos[0] - self.old_bg_box[0] - self.ext_len[2], self.pred_cpos[1] - self.old_bg_box[1] - self.ext_len[0]

        cv2.rectangle(rgb_response_map, (bg_pred_x, bg_pred_y), (bg_pred_x + self.tg_box[2], bg_pred_y + self.tg_box[3]), (255, 0, 255), 1)
        cv2.rectangle(rgb_color_map, (bg_pred_x, bg_pred_y), (bg_pred_x + self.tg_box[2], bg_pred_y + self.tg_box[3]), (255, 0, 255), 1)

        return rgb_color_map, rgb_response_map
