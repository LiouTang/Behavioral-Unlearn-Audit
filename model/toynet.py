import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyNet(nn.Module):
    def __init__(self, num_classes=4, hidden_layers=0):
        super(ToyNet, self).__init__()
        # Note that for hidden_layers > 0, the model is no longer convex
        hidden = []
        for _ in range(hidden_layers):
            hidden.append(nn.Linear(6, 6))
            hidden.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden)
        self.output = nn.Linear(6, num_classes, bias=False)

    def forward(self, x):
        x = self.feature_lift(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

    @staticmethod
    def feature_lift(X2):
        """phi(x) = (x1, x2, x1*x2, |x1|, |x2|, 1) — D = 6."""
        x1, x2 = X2[:, 0], X2[:, 1]
        ones = torch.ones_like(x1)
        return torch.stack([x1, x2, x1 * x2, torch.abs(x1), torch.abs(x2), ones], dim=1)

def ToyNet0(num_classes):
    return ToyNet(num_classes=num_classes, hidden_layers=0)

def ToyNet10(num_classes):
    return ToyNet(num_classes=num_classes, hidden_layers=10)
