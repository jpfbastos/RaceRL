import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.actor(x), dim=1)


class CriticNet(nn.Module):
    def __init__(self, n_inputs):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.critic(x).squeeze(1)