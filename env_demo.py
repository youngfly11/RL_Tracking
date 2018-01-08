# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 下午8:10
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

from env import Env


def main():

    # TODO
    # change path
    env = Env('/Users/piaozhx/tracking dataset/VOT16/', 'VOT')
    env.reset('bag')

    while True:
        reward, next_frame, done = env.step(0)
        print reward
        if done:
            break


if __name__ == '__main__':
    main()
