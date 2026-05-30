import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=600, out_dim=100, hidden_layers=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        hidden = []
        for _ in range(hidden_layers):
            hidden.append(nn.Linear(1024, 1024))
            hidden.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden(x)
        x = self.fc2(x)
        return x

def MLP4(num_classes):
    return MLP(out_dim=num_classes, hidden_layers=4)
