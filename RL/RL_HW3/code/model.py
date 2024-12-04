import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)

'''
#需要调整以实现Dueling DQN
'''
class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

class DuelingDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(DuelingDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.reshape(batch_size, -1)
        value_stream = self.value_stream(x)
        advantage_stream = self.advantage_stream(x)
        q_value = value_stream + (advantage_stream - advantage_stream.mean(dim=1, keepdim=True))
        return q_value

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)