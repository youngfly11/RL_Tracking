# -*- coding: utf-8 -*-
# @Time    : 2018/1/10 上午4:12
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import cv2
from Unsupervised_algorithm.kcftracker import KCFTracker
from Unsupervised_algorithm.hist import HistTracker
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
import os
import math

SEQ_NAME = 'basketball'

# plt.close()
# plt.ion()
fig = plt.figure()
fig.suptitle(SEQ_NAME)

ax1 = fig.add_subplot(2, 2, 1)  # gt frame axis
ax2 = fig.add_subplot(4, 4, 3)  # pred frame axis
ax3 = fig.add_subplot(4, 4, 4)  # diff frame axis
ax4 = fig.add_subplot(2, 1, 2)  # score axis
# ax5 = self.fig.add_subplot(2, 3, 6)  # labels axis

data = np.load('%s.npz' % SEQ_NAME)

# np.savez('%s.npz' % self.seq_name, predict=self.predict_list, color_map_list=self.color_map_list, response_map_list=self.response_map_list,
#          frame_idx_list=self.frame_idx_list, reward_list=self.reward_list)

# h, w = data['predict'].shape[:2]
black_img = np.zeros_like(data['predict'][0], dtype=np.uint8)
black_response_map = np.zeros_like(data['response_map_list'][0], dtype=np.uint8)
black_color_map = np.zeros_like(data['color_map_list'][0], dtype=np.uint8)

# ax1, gt frame axis
ax1.set_xlabel('predict image', fontsize=15)
ax1.set_xticks([]), ax1.set_yticks([])
ax1_img_handle = ax1.imshow(black_img, animated=True)

# ax2, pred frame axis
ax2.set_xlabel('color map', fontsize=15)
ax2.set_xticks([]), ax2.set_yticks([])
ax2_img_handle = ax2.imshow(black_response_map, animated=True)
# ax2.set_title('Testing on ' + self.seq_name, fontsize=30)

# ax3, diff frame axis
ax3.set_xlabel('response map', fontsize=15)
ax3.set_xticks([]), ax3.set_yticks([])
ax3_img_handle = ax3.imshow(black_color_map, animated=True)

# ax4, scroes frame axis
show_number = len(data['frame_idx_list'])
ax4.set_xlabel('#Frame(t)', fontsize=15)
ax4.set_ylabel('Score', fontsize=15)
ax4.set_ylim(0, 1.0)
ax4.set_xlim(0, show_number)
ax4.yaxis.set_major_locator(MultipleLocator(0.1))

ax4_line_handel, = ax4.plot([], [], animated=True)

t = 0


for t in range(len(data['predict'])):

    ax1_img_handle.set_array(data['predict'][t])
    ax2_img_handle.set_array(data['color_map_list'][t])
    ax3_img_handle.set_array(data['response_map_list'][t])
    ax4_line_handel.set_data(data['frame_idx_list'][:t + 1], data['reward_list'][:t + 1])
    # plt.show()
    print t
    name = 'results/{}'.format(t)
    plt.savefig(name)

# def generate_data(*args):
#     global t
#
#
#     ax1_img_handle.set_array(data['predict'][t])
#     ax2_img_handle.set_array(data['color_map_list'][t])
#     ax3_img_handle.set_array(data['response_map_list'][t])
#     ax4_line_handel.set_data(data['frame_idx_list'][:t + 1], data['reward_list'][:t + 1])
#
#     if t != len(data['predict']):
#         t += 1
#         print t
#     return ax1_img_handle, ax2_img_handle, ax3_img_handle, ax4_line_handel
#
# def update_func(*args):
#     while True:
#         yield generate_data()

#
# ani = animation.FuncAnimation(fig, generate_data, frames=100,interval=20, blit=True)
# ani.save('dynamic_images2.mp4', fps=30)
# # plt.show()
