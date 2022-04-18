#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lenet architecture:
 [1] http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

"""
import torch.nn as nn


# Create Architecture
class LENET(nn.Module):

    def __init__(self, num_classes):

        super(LENET, self).__init__()

        # Declare convolutional layers
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Declare fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256, 84),
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass of the model'''

        x = self.c1(x)
        x = self.c2(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
