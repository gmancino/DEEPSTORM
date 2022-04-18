#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom dataset class for the datasets used in the MLP experiments

Naming conventions are the same as in MNIST

"""

# Import files
import os
import torch
from torch.utils.data import Dataset


# Make custom class
class BinaryDataset(Dataset):
    def __init__(self, root, dataset_name, train=True):

        # Save the images
        self.data = torch.load(os.path.join(root, os.path.join(dataset_name,
                                                    dataset_name + f"_{'train' if train else 'test'}.pt")))
        self.targets = torch.load(os.path.join(root, os.path.join(dataset_name,
                                                dataset_name + f"_{'train' if train else 'test'}_labels.pt")))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        return self.data[idx], self.targets[idx]
