import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import resnet
from torch.autograd import Variable
from torch.distributions import Bernoulli
EPS = 0.003


def Fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class TrackModel(nn.Module):

    def __init__(self, pretrained=True):
        super(TrackModel, self).__init__()
        self.feature_extractor = resnet.resnet18(pretrained=pretrained).cuda()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.actor = Actor(state_dim=256, action_space=2).cuda()
        self.critic = Critic(state_dim=256, action_dim=1).cuda()
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=1).cuda()
        self.hidden_size = 256
        self.fc = nn.Linear(512, 256)

    def forward(self, imgs, hidden_prev=None):

        """
        Main model for tracking
        Args:
        ----
        - imgs: input single image 1*3*224*224
        - hidden_prev: (h0, c0)  1*1*256
        - action_np: numpy format action, just number
        - action: 1*1 cuda tensor format
        - action_sample: 1 cuda tesor format
        - value 1 cuda tensor format, output from the critic network.
        - hidden_pres: after change, Variable(h.data)
        """

        # hidden_prev: tuple, (h0,c0)
        # import  pdb; pdb.set_trace()
        state = self.feature_extractor(imgs) # 1*256
        state = self.fc(state)
        state = F.elu(state)
        state = state[None, :, :]  # change the hidden state [batch, state_dim] -> [1, batch, state_dim]
        # import  pdb; pdb.set_trace()
        _, hidden_pres = self.rnn(state, hidden_prev)
        h0 = hidden_pres[0].squeeze(0) # 1*256

        # for actor network
        # import  pdb; pdb.set_trace()

        prob, logprob = self.actor(h0)
        action_detach = prob
        m = Bernoulli(action_detach[0, 1])
        sample = m.sample()  #[1]

        # for critic network
        action = sample.unsqueeze(1) # [1, 1]
        # import  pdb; pdb.set_trace()

        value = self.critic(h0, action)

        hidden_pres = (Variable(hidden_pres[0].data), Variable(hidden_pres[1].data))
        return prob, logprob, sample, value, hidden_pres

    def init_hidden_state(self, batch_size):
        return(Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(),
               Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda())


class Critic(nn.Module):

    def __init__(self, state_dim=256, action_dim=1):
        """
        The network is to estimate the value of reward;

        Args:
        ----
        - state_dim: input hidden state dimensions, int,256
        - action_dim: input action dimension, int 1
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = Fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = Fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim,128)
        self.fca1.weight.data = Fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = Fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-EPS,EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network

        Args:
        ----
        - state:  Input state (Torch Variable : [n,state_dim] )
        - action: Input Action (Torch Variable : [n,action_dim]
        - Value:  Value function : Q(S,a) (Torch Variable : [n,1] ). The true rewards will lying in [0, 1]
        """
        s1 = F.elu(self.fcs1(state))
        s2 = F.elu(self.fcs2(s1))
        a1 = F.elu(self.fca1(action))
        x = torch.cat((s2,a1), dim=1)
        x = self.fc3(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim=256, action_space=2):

        """
        Estimate the actions
        Args:
            - state_dim: Dimension of input state (int)
            - action_dim: Dimension of output action (int)
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_space

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = Fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = Fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = Fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64, action_space)
        self.fc4.weight.data.uniform_(-EPS, EPS)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled.

        Args:
        ----
        -state: input state (torch Variable: [n, state_dim])
        -prob, logprob: output probability and logsoftmax; [n, action_dim]
        """
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        prob = self.softmax(x)
        logprob = self.logsoftmax(x)
        return prob, logprob


if __name__=='__main__':
    critic = Critic()
    state = Variable(torch.ones(1, 256))
    action = Variable(torch.ones(1).unsqueeze(0))

    value = critic(state, action)
    print value.size()
