import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.misc import AverageMeter
from Unsupervised_algorithm.env import Env
from utils.dataloader import ReadSingleImage
import numpy as np


def test(args, model, video_val=None):
    reward_avg = AverageMeter()
    loss_avg = AverageMeter()
    value_loss_avg = AverageMeter()
    policy_loss_avg = AverageMeter()

    root_dir = '/home/youngfly/DL_project/RL_Tracking/dataset/VOT'
    data_type = 'VOT'

    model.eval()
    env = Env(seqs_path=root_dir,
              data_set_type=data_type,
              save_path='/home/youngfly/DL_project/RL_Tracking/dataset/Result/VOT')

    for video_name in video_val:

        # env = Env(seqs_path=root_dir, data_set_type=data_type, save_path='dataset/Result/VOT')
        actions = []
        rewards = []
        values = []
        entropies = []
        logprobs = []

        # reset for new video
        observation1, observation2 = env.reset(video_name)
        img1 = ReadSingleImage(observation2)
        img1 = Variable(img1).cuda()

        hidden_prev = model.init_hidden_state(batch_size=1)  # variable cuda tensor
        _, _, _, _, hidden_pres = model(imgs=img1, hidden_prev=hidden_prev)

        # for loop init parameter
        hidden_prev = hidden_pres
        observation = observation2
        FLAG = 1
        i = 2
        while FLAG:
            img = ReadSingleImage(observation)
            img = Variable(img).cuda()

            action_prob, action_logprob, action_sample, value, hidden_pres = model(imgs=img, hidden_prev=hidden_prev)

            entropy = -(action_logprob * action_prob).sum(1, keepdim=True)
            entropies.append(entropy)

            actions.append(action_sample.long())  # list, Variable cuda inner
            action_np = action_sample.data.cpu().numpy()

            sample = Variable(torch.LongTensor(action_np).cuda()).unsqueeze(0)

            hidden_prev = hidden_pres
            logprob = action_logprob.gather(1, sample)
            logprobs.append(logprob)

            reward, new_observation, done = env.step(action=action_np)
            env.show_all()
            # env.show_tracking_result()

            print('test:','frame:%d'%(i), 'Action:%d'%action_np[0], 'rewards:%.6f'%reward, 'probability:%.6f%.6f'%(action_prob.data.cpu().numpy()[0, 0],
                  action_prob.data.cpu().numpy()[0, 1]))
            i = i+1
            rewards.append(reward)  # just list
            values.append(value)  # list, Variable cuda inner
            observation = new_observation

            if done:
                FLAG = 0

        num_seqs = len(rewards)
        running_add = Variable(torch.FloatTensor([0])).cuda()
        value_loss = 0
        policy_loss = 0
        gae = torch.FloatTensor([0]).cuda()
        values.append(running_add)
        for i in reversed(range(len(rewards))):
            running_add = args.gamma * running_add + rewards[i]
            advantage = running_add - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - logprobs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        # value_loss = value_loss / num_seqs
        # policy_loss = policy_loss / num_seqs
        #
        # values.append(running_add)
        # for i in reversed(range(len(rewards))):
        #     running_add = args.gamma * running_add + rewards[i]
        #     advantage = running_add - values[i]
        #     value_loss = value_loss + 0.5 * advantage.pow(2)
        #     policy_loss = policy_loss - logprobs[i] * advantage - args.entropy_coef * entropies[i]
        #
        value_loss = value_loss / num_seqs
        policy_loss = policy_loss / num_seqs

        loss = args.value_loss_coef * value_loss + policy_loss

        print(video_name, 'rewards:%.6f'%np.mean(rewards), 'loss:%.6f'%loss.data[0], 'value_loss:%6f' %
              value_loss.data[0], 'policy_loss:%.6f' % policy_loss.data[0])

        # update the loss
        loss_avg.update(loss.data.cpu().numpy())
        value_loss_avg.update(value_loss.data.cpu().numpy())
        policy_loss_avg.update(policy_loss.data.cpu().numpy())
        reward_avg.update(np.mean(rewards))

    return reward_avg.avg, loss_avg.avg, value_loss_avg.avg, policy_loss_avg.avg