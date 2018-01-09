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
from model import tracking, tracking_v1
import argparse
# import utils.vis_gradient as viz

parser = argparse.ArgumentParser(description='RL_Tracking')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max-grad-norm', type=float, default=20,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='number of forward epochs in RL Tracking  (default: 50)')

parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')

parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.1)')
parser.add_argument('--env-name', default='Tracking',
                    help='environment to train on Visdom')


# print the parameter's gradients
def hook_print(grad):
    print(grad)



def main():

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if not osp.exists(osp.join('checkpoints', 'Tracking')):
        os.makedirs(osp.join('checkpoints', 'Tracking'))
    vis = Dashboard(server='http://localhost', port=8097, env='Tracking')

    model = tracking_v1.TrackModel(pretrained=True)
    model = model.cuda()
    # grad = viz.create_viz('main', model, env = 'Tracking')
    # grad.regis_weight_ratio_plot('critic.fc2', 'weight', 'g/w')

    # feature_extractor network, the same learning rate
    # optimizer = torch.optim.Adam([{'params': model.fc.parameters()},
    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters()},
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

    model.critic.fc2.register_backward_hook(lambda module, grad_input, grad_output: grad_output)
    model.actor.fc1.weight.register_hook(hook_print)
    for epoch in range(args.num_epochs):

        reward_train, loss_train, loss1_train, loss2_train = train(args=args,
                                                                   model=model,
                                                                   optimizer=optimizer,
                                                                   video_train=video_train)
        reward_val, loss_val, loss1_val, loss2_val = test(args=args,
                                                          model=model,
                                                          video_val=video_val)

        train_loss[epoch] = loss_train[0, 0]
        train_loss1[epoch] = loss1_train[0, 0]
        train_loss2[epoch] = loss2_train[0, 0]
        train_reward[epoch] = reward_train
        val_loss[epoch] = loss_val[0, 0]
        val_loss1[epoch] = loss1_val[0, 0]
        val_loss2[epoch] = loss2_val[0, 0]
        val_reward[epoch]= reward_val

        # for visualization
        vis.draw(train_data=train_loss, val_data=val_loss, datatype='loss')
        vis.draw(train_data=train_loss1, val_data=val_loss1,datatype='Loss1')
        vis.draw(train_data=train_loss2, val_data=val_loss2, datatype='Loss2')
        vis.draw(train_data=train_reward, val_data=val_reward, datatype='rewards')

        print ('Train', 'epoch:', epoch, 'rewards:{%.6f}'%reward_train, 'loss:{%.6f}'%loss_train),
        print ('validation', 'epoch:', epoch, 'rewards:{%.6f}'%reward_val, 'loss:{%.6f}'%loss_val),

        if best_loss > loss_val[0, 0]:
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