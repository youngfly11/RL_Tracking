# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 下午8:10
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

from env import Env
import random


def main():
    env = Env('/Users/piaozhx/tracking dataset/VOT16/', 'VOT', 'data')
    env.reset('bag')

    f_num = len(env.gt_rects)

    action_list = []
    for i in xrange(f_num):
        action_list.append(random.randint(0,1))

    for i in action_list:
        reward, next_frame, done = env.step(i)
        env.show_all()
        print reward
        if done:
            break


if __name__ == '__main__':
    main()
