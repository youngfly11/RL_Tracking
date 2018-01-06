from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
import train
from utils.misc import AverageMeter, save_checkpoint
from utils import env
from utils.dataloader import get_load, ReadSingleImage
from utils.video import video_train, video_val

from utils.misc import AverageMeter
from utils.visualize import Dashboard
import os.path as osp

os.makedirs(osp.join('checkpoints','Tracking'))
vis = Dashboard(port=8099,server='https://127.0.0.1', env='Tracking')


EPOCHS = 50
NUM_VIDEO= 60

print ' NUMBER OF VIDEO :- ', NUM_VIDEO


def main():

    trainer = train.Trainer(state_dim=256, action_space=2).cuda()
    best_loss = 100
    train_loss1 = {}
    train_loss2 = {}
    train_loss = {}
    val_loss = {}
    val_loss1 = {}
    val_loss2 = {}
    train_reward = {}
    val_reward = {}
    for epoch in range(EPOCHS):
        reward_train, loss_train, loss1_train, loss2_train = train_epoch(trainer=trainer, video_train=video_train)
        reward_val, loss_val, loss1_val, loss2_val = val_epoch(trainer=trainer, video_val=video_val)

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
                'state_dict': trainer.model.state_dict(),
                'best_loss1': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_rewards': train_reward,
                'val__rewards': val_reward,
            }, is_best,
                filename='epoch_{}.pth.tar'.format(epoch + 1),
                dir=os.path.join('checkpoints', 'Tracking'), epoch=epoch)


def train_epoch(trainer=None, video_train=None):


    # TODO
    # env.reset(), return fisrt observation
    reward_avg = AverageMeter()
    loss_avg = AverageMeter()
    loss1_avg = AverageMeter()
    loss2_avg = AverageMeter()
    for video_name in video_train:


        actions = []
        rewards = []
        observations = []
        hidden_for_train = None
        observation1, observation2 = env.reset(video_name)  # array
        img1 = ReadSingleImage(img=observation1)
        img1 = Variable(img1).cuda()
        hidden_prev = trainer.model.init_hidden_state(batch_size=1)  # cuda tensor
        first_action, hidden_prev = trainer.get_exploration_action(img=img1, hidden_prev=hidden_prev)
        hidden_for_train = hidden_prev
        observation = observation2
        flag = 1
        while flag:
            observations.append(observation)
            img = ReadSingleImage(img=observation)
            img = Variable(img).cuda()
            action = trainer.get_exploration_action(img=img, hidden_prev=hidden_prev).detach()

            action = action.data.cpu.numpy()[0]  ## change cuda tensor to numpy
            actions.append(action)
            new_observation, reward, done = env.step(action)

            actions.append(action)
            rewards.append(reward)  # list for numpy

            observation = new_observation

            if done:
                flag = 0

                reward_avg.update(np.mean(np.array(rewards)))

                rewards = trainer.reward_to_go(rewards=rewards)
                train_loader = get_load(imgs=observations, actions=actions, rewards=rewards)
                loss, loss1, loss2 = trainer.optimize(train_loader=train_loader, hidden_prev=hidden_for_train)
                print(video_name, 'rewards:', np.mean(np.array(rewards)), 'loss:', loss)

                loss_avg.update(loss.cpu().numpy())
                loss1_avg.update(loss1.cpu().numpy())
                loss2_avg.update(loss2.cpu().numpy())


    return reward_avg.avg, loss_avg.avg, loss1_avg.avg, loss2_avg.avg


def val_epoch(trainer=None, video_val=None):
    # TODO
    # env.reset(), return fisrt observation
    reward_avg = AverageMeter()
    loss_avg = AverageMeter()
    loss1_avg = AverageMeter()
    loss2_avg = AverageMeter()

    for video_name in video_val:

        actions = []
        rewards = []
        observations = []
        observation1, observation2 = env.reset(video_name)  # array
        img1 = ReadSingleImage(img=observation1)
        img1 = Variable(img1).cuda()
        hidden_prev = trainer.model.init_hidden_state(batch_size=1)  # cuda tensor
        first_action, hidden_prev = trainer.get_exploration_action(img=img1, hidden_prev=hidden_prev)
        hidden_for_train = hidden_prev
        observation = observation2
        flag = 1
        while flag:
            observations.append(observation)
            img = ReadSingleImage(img=observation)
            img = Variable(img).cuda()
            action = trainer.get_exploration_action(img=img, hidden_prev=hidden_prev).detach()

            action = action.data.cpu.numpy()[0]  ## change cuda tensor to numpy
            actions.append(action)
            new_observation, reward, done = env.step(action)

            actions.append(action)
            rewards.append(reward)  # list for numpy

            observation = new_observation

            if done:
                flag = 0

                reward_avg.update(np.mean(np.array(rewards)))

                rewards = trainer.reward_to_go(rewards=rewards)
                train_loader = get_load(imgs=observations, actions=actions, rewards=rewards)
                loss, loss1, loss2 = trainer.optimize(train_loader=train_loader, hidden_prev=hidden_for_train, is_train=False)
                print(video_name, 'rewards:', np.mean(np.array(rewards)), 'loss:', loss)

                loss_avg.update(loss.cpu().numpy())
                loss1_avg.update(loss1.cpu().numpy())
                loss2_avg.update(loss2.cpu().numpy())

    return reward_avg.avg, loss_avg.avg, loss1_avg.avg, loss2_avg.avg