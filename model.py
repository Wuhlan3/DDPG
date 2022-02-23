import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, dim_s, dim_a):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(dim_s + dim_a, 512)
        self.l2 = nn.Linear(512 , 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Actor(nn.Module):
    def __init__(self, dim_s, dim_a, max_a):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(dim_s, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, dim_a)
        self.max_a = max_a

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_a * torch.tanh(self.l3(x))
        return x