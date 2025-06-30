import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import math

# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)  # pay attention to add training=self.training
        x = self.fc2(x)
        return x


        
