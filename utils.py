# @author Dylan Goetting
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    # Set up memory storage
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # Simple neural net consisting only of fully connected linear layers. Input is a 1x1 tensor and output is a 1x2 tensor representing the expected discounted value of choosing each action
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Distribution(object):

    def __init__(self, dist, mean, sd):
        self.dist = dist
        self.mean = mean
        self.sd = sd
    
    def sample(self):
        return self.dist.rvs(loc=self.mean, scale=self.sd)


class CriticNet(nn.Module):

    def __init__(self, input_size, output_size=1):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)

        self.fc5 = nn.Linear(input_size, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, output_size)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        q2 = F.relu(self.fc5(x))
        q2 = F.relu(self.fc6(q2))
        q2 = F.relu(self.fc7(q2))
        q2 = self.fc8(q2)

        return q1, q2

    def q1(self, states, actions):
        x = torch.cat((states, actions), 1)
        q = F.relu(self.fc1(x))
        q = F.relu(self.fc2(q))
        q = F.relu(self.fc3(q))

        return self.fc4(q)


class ActorNet(nn.Module):

    def __init__(self, input_size, output_size=1, min_out=0, max_out=1 ):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)
        self.min_out = min_out
        self.max_out = max_out

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        
        dev = (self.max_out - self.min_out)/2
        mid = (self.max_out + self.min_out)/2
        x = mid + dev*x        

        return x
