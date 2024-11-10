# AngleEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class AngleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AngleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
