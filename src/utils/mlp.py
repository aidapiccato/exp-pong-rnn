"""Multi-layer perceptron"""

import torch.nn as nn 
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    