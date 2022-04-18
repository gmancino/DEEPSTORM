#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2 Layer NN

"""
import torch.nn as nn


# Make main class
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):

        super(MLP, self).__init__()

        # Declare convolutional layers
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass of the model'''

        x = self.l1(x)

        return x