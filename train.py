from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.tracking import TrackModel
import numpy as np
from torch.distributions import Bernoulli
import math
import utils
from utils.misc import AverageMeter

BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99
TAU = 0.001


class Trainer:

    def __init__(self, state_dim=256, action_space=2):
        """
        Args:
            - state_dim: Dimensions of state (int)
            - action_dim: Dimension of action (int)
        """
        self.state_dim = state_dim
        self.action_space = action_space
        self.model = TrackModel(pretrained=True).cuda()

        self.optimizer = torch.optim.Adam([{'params': self.model.feature_extractor.parameters(), 'lr': 0.5*LEARNING_RATE},
                                           {'params': self.model.rnn.parameters()},
                                           {'params': self.model.actor.parameters()},
                                           {'params': self.model.critic.parameters()}],
                                           lr=LEARNING_RATE)
        self.loss1 = torch.nn.NLLLoss().cuda()
        self.loss2 = torch.nn.MSELoss().cuda()

    def get_exploration_action(self, img, hidden_prev):
        """
        Play mode to get actions
        Args:
        ----
        - img: input is the single image(1*3*height*weight)
        - action, sample from the probability. p0 = action[0], p1 = action[1], 1*2
        """

        action, hidden_pres = self.model.forward(img, action=None, is_play=True, hidden_prev=hidden_prev)
        action = action.detach() # stop gradients from actions
        m = Bernoulli(action[1])
        action_final = m.sample()  # cuda tensor
        action = action_final.data.cpu().numpy()[0]  # change cuda tensor to numpy
        hidden = (Variable(hidden_pres[0].data, Variable(hidden_pres[1].data)))
        return action, hidden

    def optimize(self, train_loader, hidden_pres, is_train=True):
        """
        optimize the RL network.
        imgs: list for all imgs, np.array format() 224*224
        rewards: cuda tensor format. list. reward to go
        action: cuda tensor format.list
        """

        loss_avg = AverageMeter()
        loss1_avg = AverageMeter()
        loss2_avg = AverageMeter()

        for idx, sample in enumerate(train_loader):

            imgs, actions, rewards = sample['images'], sample['actions'], sample['rewards']
            if is_train:
                imgs = Variable(imgs).cuda()
            else:
                imgs = Variable(imgs, volatile=True).cuda()
            actions = Variable(actions).cuda()
            rewards = Variable(rewards).cuda()

            # b*2, b*2, 1*b*256, b*1
            actions_prob, action_logprob, hidden_pres, value = self.model.forward(img=imgs,
                                                                                  action=actions,
                                                                                  hidden_prev=hidden_pres,
                                                                                  is_play=False)
            loss1 = self.loss1(action_logprob, actions)
            loss2 = self.loss2(rewards, value)
            loss = loss1+loss2
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss = loss.data.cpu().numpy()[0]
            loss1 = loss1.data.cpu().numpy()[0]
            loss2 = loss2.data.cpu().numpy()[0]
            loss_avg.update(loss)
            loss1_avg.update(loss1)
            loss2_avg.update(loss2)

        return loss_avg.avg, loss1_avg.avg, loss2_avg.avg

    def reward_to_go(self, rewards):
        running_add = 0

        for t in reversed(range(0, len(rewards))):

            running_add = running_add + GAMMA* rewards[t]
            rewards[t] = running_add

        # mean1 = [reward.data.cpu().numpy() for reward in rewards]
        # mean1 = np.array(mean1)
        # mean = np.mean(mean1)
        # std = np.std(mean1)
        # rewards -= mean
        # rewards /= std
        return rewards
