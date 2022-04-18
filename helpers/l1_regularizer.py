#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform L_1- Regularization on LIST of PyTorch Tensors

"""

# Import files
import torch

class L1:
    '''
    L_1-Regularization for PyTorch

    '''

    def __init__(self, device):
        '''

        :param regularizer: float value greater than or equal to 0.0
        '''

        # Save regularizer and device
        self.device = device

        # Save thresh-holder; gives element-wise max of zero and input
        self.max_0 = torch.nn.Threshold(threshold=0.0, value=0.0)

    def forward(self, list_of_tensors, reg):
        '''Perform regularization'''

        # Save computation if regularizer is 0
        if reg == 0.0:

            return list_of_tensors

        else:
            # Loop over all of the tensors
            for i, t in enumerate(list_of_tensors):

                # Modify current tensors
                list_of_tensors[i] = torch.sign(t).to(self.device) * self.max_0(torch.abs(t) - reg).to(self.device)

            return list_of_tensors

    def number_non_zeros(self, list_of_tensors):
        '''Count the number of non-zero entries'''

        # Save values
        nnz = 0
        total_params = 0

        # Loop over the parameters
        for _, t in enumerate(list_of_tensors):

            total_params += torch.prod(torch.tensor(t.shape)).item()
            nnz += (torch.abs(t) > 1e-6).sum().item()

        # Return both the count and the ratio
        return nnz, (nnz / total_params)