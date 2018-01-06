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


BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99
TAU = 0.001


class Trainer:

    def __init__(self, state_dim=256, action_space=2):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_space
        self.model = TrackModel(pretrained=True).cuda()

        self.optimizer = torch.optim.Adam([{'params': self.model.feature_extractor.parameters(), 'lr': 0.1*LEARNING_RATE},
                                                 {'params': self.model.rnn.parameters()},
                                                 {'params': self.model.actor.parameters()},
                                                 {'params': self.model.critic.parameters()}],
                                                lr=LEARNING_RATE)
        self.loss1 = torch.nn.NLLLoss().cuda()
        self.loss2 = torch.nn.MSELoss().cuda()


    def get_exploration_action(self, img, hidden_prev):
        """
        Gets the action from the target actor network.
        Args:
        ----
        - img: input is the single image(1*3*height*weight)
        - action, sample from the probability. p0 = action[0], p1 = action[1]
        """

        action, hidden_pres = self.model.forward(img, action=None, is_play=True, hidden_prev=hidden_prev)
        action = action.detach()
        m = Bernoulli(action[1])
        action_final = m.sample()  # cuda tensor
        hidden = (Variable(hidden_pres[0], Variable(hidden_pres[1])))
        return action_final, hidden

    def optimize(self, train_loader, hidden_pres, is_train=True):
        """
        optimize the RL network.
        imgs: list for all imgs, np.array format() 224*224
        rewards: cuda tensor format. list. reward to go
        action: cuda tensor format.list
        """

        # TODO
        # the first frame need to deal
        ## from the outside


        for idx, sample in enumerate(train_loader):

            imgs, actions, rewards = sample['images'], sample['actions'], sample['rewards']
            imgs = Variable(imgs).cuda()
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

        return loss.data[0], loss1.data[0], loss2.data[0]

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


    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print 'Models saved successfully'


    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print 'Models loaded succesfully'
