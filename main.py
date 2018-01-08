from __future__ import division
import numpy as np
import torch
import os
from train import train
from test import test
from utils.misc import save_checkpoint
from utils.video import video_train, video_val
from utils.visualize import Dashboard
import os.path as osp
from model import tracking
import argparse

parser = argparse.ArgumentParser(description='RL_Tracking')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max-grad-norm', type=float, default=5,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='number of forward epochs in RL Tracking  (default: 50)')

parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')

parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--env-name', default='Tracking',
                    help='environment to train on Visdom')


def main():

    args = parser.parse_args()

    if not osp.exists(osp.join('checkpoints', 'Tracking')):
        os.makedirs(osp.join('checkpoints', 'Tracking'))
    vis = Dashboard(port=8099, server='https://127.0.0.1', env='Tracking')

    model = tracking.TrackModel(pretrained=True)
    model = model.cuda()

    optimizer = torch.optim.Adam([{'params':model.feature_extractor.parameters(),'lr': 0.1 * args.lr},
                                  {'params':model.actor.parameters()},
                                  {'params':model.critic.parameters()},
                                  {'params':model.rnn.parameters()}],
                                 lr = args.lr)
    best_loss = 100
    train_loss1 = {}
    train_loss2 = {}
    train_loss = {}
    val_loss = {}
    val_loss1 = {}
    val_loss2 = {}
    train_reward = {}
    val_reward = {}

    for epoch in range(args.num_epochs):
        reward_train, loss_train, loss1_train, loss2_train = train(args=args, model=model, optimizer=optimizer, video_train=video_train)
        reward_val, loss_val, loss1_val, loss2_val = test(args=args, model=model, video_val=video_val)

        train_loss[epoch] = loss_train
        train_loss1[epoch] = loss1_train
        train_loss2[epoch] = loss2_train
        train_reward[epoch] = reward_train
        val_loss[epoch] = loss_val
        val_loss1[epoch] = loss1_val
        val_loss2[epoch] = loss2_val
        val_reward[epoch]= reward_val

        vis.draw(train_data=train_loss, val_data=val_loss,datatype='loss')
        vis.draw(train_data=train_loss1, val_data=val_loss1,datatype='Loss1')
        vis.draw(train_data=train_loss2, val_data=val_loss2, datatype='Loss2')
        vis.draw(train_data=reward_train, val_data=reward_val,datatype='rewards')

        print ('Train', 'epoch:', epoch, 'rewards', reward_train, 'loss:', loss_train),
        print ('validation', 'epoch:', epoch, 'rewards', reward_val,'loss:', loss_val),

        if best_loss > loss_val:
            best_loss = loss_val
            is_best = True
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss1': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_rewards': train_reward,
                'val_rewards': val_reward,
            }, is_best,
                filename='epoch_{}.pth.tar'.format(epoch + 1),
                dir=os.path.join('checkpoints', 'Tracking'), epoch=epoch)


if __name__ == '__main__':

    np.random.seed(10)
    torch.manual_seed(10)
    main()