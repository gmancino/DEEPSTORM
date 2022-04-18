#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test SPPDM
    [1] https://arxiv.org/pdf/2011.05082.pdf

"""

# Import packages
from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import numpy
import math
from mpi4py import MPI
from torchvision import datasets, transforms

# Import custom classes
from models.mlp import MLP
from models.lenet import LENET
from helpers.l1_regularizer import L1
from helpers.replace_weights import Opt
from helpers.custom_data_loader import BinaryDataset

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# Declare new method
class SPPDM:
    '''
    Class for solving decentralized nonconvex consensus problems.

    :param: local_params = DICT of parameters for training
    :param: mixing_matrix = NxN torch float containing weights for communication
    :param: training_data = torch.utils.data.Dataloader
    :param: init_weights = LIST of NUMPY arrays containing initial weights for the network
    '''

    def __init__(self, local_params, mixing_matrix, training_data, init_weights):

        # Get the information about neighbor communication:
        # First, we extract the number of nodes and double check
        # this value is the same as the size of the MPI world
        # Second, we extract thr row of the mixing matrix corresponding to this agent
        # and save the weights
        self.mixing_matrix = mixing_matrix.float()
        self.num_nodes = self.mixing_matrix.shape[0]
        if self.num_nodes != size:
            sys.exit(f"Cannot match MPI size {size} with mixing matrix of shape {self.num_nodes}. ")
        # Get degree and adjacency matrix
        self.degree = torch.diag((self.mixing_matrix != 0).sum(0) - 1)
        self.adjacency = (self.mixing_matrix != 0).long() - torch.eye(self.num_nodes)

        self.peers = torch.where(self.mixing_matrix[rank, :] != 0)[0].tolist()
        self.peers.remove(rank)
        self.peer_weights = self.adjacency[rank, self.peers].tolist()
        self.my_weight = self.degree[rank, rank].item()

        ##################################################
        # Parse the training parameters:
        #
        # alpha = learning rate (FLOAT)
        # c = learning rate (FLOAT)
        # gamma = learning rate (FLOAT)
        # kappa = learning rate (FLOAT)
        # beta = learning rate (FLOAT)
        # mini_batch = batch size (INT)
        # momentum = 'none' or 'nesterov' or 'constant' (STR)
        # l1 = regularization coefficient (FLOAT)
        # report = how often to report stationarity, test acc, etc. (INT)
        ##################################################
        if 'alpha' in local_params:
            self.alpha = local_params['alpha']
        else:
            self.alpha = 1e-2
        if 'c' in local_params:
            self.c = local_params['c']
        else:
            self.c = 1
        if 'gamma' in local_params:
            self.gamma = local_params['gamma']
        else:
            self.gamma = 3
        if 'kappa' in local_params:
            self.kappa = local_params['kappa']
        else:
            self.kappa = 1e-1
        if 'beta' in local_params:
            self.beta = local_params['beta']
        else:
            self.beta = 0.9
        if 'mini_batch' in local_params:
            self.mini_batch = int(local_params['mini_batch'])
        else:
            self.mini_batch = 128
        if 'l1' in local_params:
            self.l1 = local_params['l1']
        else:
            self.l1 = 0.0
        if 'momentum' in local_params:
            self.momentum = local_params['momentum']
        else:
            self.momentum = 'none'
        if 'report' in local_params:
            self.report = local_params['report']
        else:
            self.report = 100

        # Compute phi
        self.phi = torch.tensor(self.gamma + 2 * self.c * self.my_weight + self.kappa).float()

        # Get the CUDA device and save the data loader to be easily reference later
        self.device = torch.device(f'cuda:{rank % size}')
        self.data_loader = training_data

        # Initialize the models
        # We either have the MLP or we have LENET
        if args.data in ['a9a', 'miniboone']:
            self.model = MLP(self.data_loader.dataset.data.shape[1], 64, 2).to(self.device)

        elif args.data == 'mnist':
            self.model = LENET(10).to(self.device)

        else:
            sys.exit(f"[ERROR] To use a new dataset/architecture, add the dataset to the data folder and incorporate the"
                     f"model here using \'self.model = <your_model>.to(self.device)\'.")

        # Initialize the updating weights rule and the training loss function
        self.replace_weights = Opt(self.model.parameters(), lr=0.1)
        self.training_loss_function = torch.nn.NLLLoss(reduction='mean')

        # Initialize the l1 regularizer
        self.regularizer = L1(self.device)

        # Load the variables - Can change this value as specified in the ReadMe
        self.weights = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
        self.old_weights = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
        self.dual = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        self.S = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
        self.Z = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]
        self.comm_var = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]

        # Save number of parameters
        self.num_params = len(self.weights)

        # Do one communication
        self.grads = self.get_full_grads(self.weights)
        self.old_grads = [self.grads[k].detach() for k in range(len(self.grads))]
        _ = self.communicate_with_neighbors()
        self.weights_onehalf = [
            ((self.gamma + self.c * self.degree[rank, rank] + self.kappa) / (self.phi)) * self.weights[k]
            + (self.c / self.phi) * self.comm_var[k] - (1 / self.phi) * self.grads[k]
            for k in range(self.num_params)]

        # Allocate space for relevant report values: consensus, gradient,
        # iterate norm, number non-zeros, training/testing acc, compute time, etc.
        self.consensus_violation = []
        self.norm_hist = []
        self.total_optimality = []
        self.iterate_norm_hist = []
        self.nnz_at_avg = []
        self.avg_nnz = []
        self.testing_loss = []
        self.testing_accuracy = []
        self.training_loss = []
        self.training_accuracy = []
        self.testing_loss_local = []
        self.testing_accuracy_local = []
        self.training_loss_local = []
        self.training_accuracy_local = []
        self.compute_time = []
        self.communication_time = []
        self.total_time = []

    def solve(self, outer_iterations, training_data_full_sample, testing_data):
        '''Solve the global problem'''

        # Barrier
        comm.Barrier()

        ##################################################
        # Save initial errors for fair comparison across methods
        avg_weights = self.get_average_param(self.weights)
        cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights, self.weights,
                                                                                            training_data_full_sample)
        self.consensus_violation.append(cons)
        self.norm_hist.append(norm)
        self.total_optimality.append(total)
        self.iterate_norm_hist.append(var_norm)
        self.nnz_at_avg.append(nnz_at_avg)
        self.avg_nnz.append(avg_nnz)

        # TEST ACCURACY ON TRAINING SET
        train_loss, train_acc = self.test(avg_weights, self.data_loader)
        self.training_loss.append(train_loss)
        self.training_accuracy.append(train_acc)

        # TEST ACCURACY ON TEST SET
        test_loss, test_acc = self.test(avg_weights, testing_data)
        self.testing_loss.append(test_loss)
        self.testing_accuracy.append(test_acc)

        # TEST ACCURACY ON TRAINING SET AT LOCAL
        train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
        self.training_loss_local.append(train_loss_local)
        self.training_accuracy_local.append(train_acc_local)

        # TEST ACCURACY ON TEST SET AT LOCAL
        test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
        self.testing_loss_local.append(test_loss_local)
        self.testing_accuracy_local.append(test_acc_local)
        ##################################################

        # Time the entire algorithm
        t0 = time.time()

        # Do first update
        self.weights = self.regularizer.forward(self.weights_onehalf, self.l1 / self.alpha)

        # Initialize momentum
        theta_old = 1

        # Barrier communication at beginning of run to force agents to start at the same time
        comm.Barrier()

        # Loop over algorithm updates
        for i in range(outer_iterations):

            ##################################################
            # Perform the algorithm updates
            # Perform communication
            comm.Barrier()
            comm_time = self.communicate_with_neighbors()
            comm.Barrier()

            # TIME THIS EPOCH
            time_i = time.time()

            # Update momentum
            theta = (1 + math.sqrt(1 + 4 * theta_old ** 2)) / 2
            if self.momentum == 'none':
                eta = 0.0
            elif self.momentum == 'nesterov':
                eta = (theta_old - 1) / theta
            else:  # constant momentum, use 0.8 from paper
                eta = 0.8
            theta_old = theta
            prev_weights = [self.weights[k].detach().clone() for k in range(self.num_params)]

            # Update S and gradient
            self.dual = [self.dual[k] + self.alpha * (self.my_weight * self.weights[k] - self.comm_var[k])
                         for k in range(self.num_params)]
            self.S = [self.weights[k] + eta * (self.weights[k] - self.old_weights[k]) for k in
                      range(self.num_params)]
            self.grads = self.get_grads(self.S)

            # Update X+1/2, X, and Z
            self.weights_onehalf = [(1 / self.phi) * (self.gamma * self.S[k] +
                                                      self.c * (self.my_weight * self.weights[k] + self.comm_var[k])
                                                      + self.kappa * self.Z[k] - self.grads[k] - self.dual[k])
                                    for k in range(self.num_params)]
            self.weights = self.regularizer.forward(self.weights_onehalf, self.l1 / self.phi)
            self.Z = [self.Z[k] + self.beta * (self.weights[k] - self.Z[k]) for k in range(self.num_params)]

            self.old_weights = [prev_weights[k].detach().clone() for k in range(self.num_params)]

            # END TIME
            time_i_end = time.time()
            comp_time = round(time_i_end - time_i, 4)
            ##################################################

            # Barrier at the end of update for extreme safety
            comm.Barrier()

            # Save values at report interval
            if i % self.report == 0:

                # Save the first errors using the average value - so all agents are compared fairly
                avg_weights = self.get_average_param(self.weights)
                cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights,
                                                                                                    self.weights,
                                                                                                    training_data_full_sample)
                self.consensus_violation.append(cons)
                self.norm_hist.append(norm)
                self.total_optimality.append(total)
                self.iterate_norm_hist.append(var_norm)
                self.nnz_at_avg.append(nnz_at_avg)
                self.avg_nnz.append(avg_nnz)

                # TEST ACCURACY ON TRAINING SET
                train_loss, train_acc = self.test(avg_weights, self.data_loader)
                self.training_loss.append(train_loss)
                self.training_accuracy.append(train_acc)

                # TEST ACCURACY ON TEST SET
                test_loss, test_acc = self.test(avg_weights, testing_data)
                self.testing_loss.append(test_loss)
                self.testing_accuracy.append(test_acc)

                # TEST ACCURACY ON TRAINING SET AT LOCAL
                train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
                self.training_loss_local.append(train_loss_local)
                self.training_accuracy_local.append(train_acc_local)

                # TEST ACCURACY ON TEST SET AT LOCAL
                test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
                self.testing_loss_local.append(test_loss_local)
                self.testing_accuracy_local.append(test_acc_local)

                # Print relevant information
                if rank == 0:
                    # First iteration, print headings, then print the values
                    if i == 0:
                        print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format("Iteration", "Epoch",
                                                                                                  "Stationarity",
                                                                                                  "Train (L / A)",
                                                                                                  "Test (L / A)",
                                                                                                  "Train (L / A) L",
                                                                                                  "Test (L / A) L",
                                                                                                  "Avg Density",
                                                                                                  "Time"))
                    print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format(i,
                                                                     round((i * self.mini_batch) / (self.data_loader.dataset.data.shape[0] // size), 2),
                                                                     round(total, 4),
                                                                     f"{round(train_loss, 4)} / {round(train_acc, 2)}",
                                                                     f"{round(test_loss, 4)} / {round(test_acc, 2)}",
                                                                     f"{round(train_loss_local, 4)} / {round(train_acc_local, 2)}",
                                                                     f"{round(test_loss_local, 4)} / {round(test_acc_local, 2)}",
                                                                     round(avg_nnz, 6),
                                                                     round(time.time() - t0, 1)))

            # Append timing information for each iteration
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)
            self.total_time.append(comp_time + comm_time)

        ##################################################
        # End total training time
        t1 = time.time() - t0
        if rank == 0:
            closing_statement = f' Training finished '
            print('\n' + closing_statement.center(50, '-'))
            print(f'[TOTAL TIME] {round(t1, 2)}')

        # Return the training time
        return t1

    def communicate_with_neighbors(self):

        # TIME IT
        time0 = MPI.Wtime()

        # ----- LOOP OVER PARAMETERS ----- #
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.weights[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.weights[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            self.comm_var[pa] = torch.zeros(self.weights[pa].shape).to(self.device)

            # Update global variables
            for ind in range(len(self.peers)):
                self.comm_var[pa] += (torch.tensor(recv_data[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)

    def get_average_param(self, list_of_params):
        '''Perform ALLREDUCE of neighbor parameters'''

        # Save information to blank list
        output_list_of_parameters = [None] * len(list_of_params)

        # Loop over the parameters
        for pa in range(self.num_params):

            # Prep send and receive to be numpy arrays
            send_data = list_of_params[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=(list_of_params[pa].shape), dtype=numpy.float32)

            # Barriers and note that the allreduce operations is summation!
            comm.Barrier()
            comm.Allreduce(send_data, recv_data)
            comm.Barrier()

            # Save information by dividing by number of agents and converting to tensor
            output_list_of_parameters[pa] = (1 / self.num_nodes) * torch.tensor(recv_data).to(self.device)

        return output_list_of_parameters

    def get_grads(self, current_weights):
        '''Get a local gradient'''

        # Update parameters
        self.replace_weights.step(current_weights, self.device)

        # Set model to training mode
        self.model.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Zero out gradients
            self.replace_weights.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Forward pass of the model
            out = self.model(data)
            loss = (1 / self.num_nodes) * self.training_loss_function(out, target)

            # Compute the gradients
            loss.backward()

            # Return sample gradient
            return [p.grad.data.detach().clone().to(self.device) for p in self.model.parameters()]

    def get_full_grads(self, current_weights):
        '''Get a local gradient'''

        # Update parameters
        self.replace_weights.step(current_weights, self.device)

        # Set model to training mode
        self.model.train()
        grads = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Zero out gradients
            self.replace_weights.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Forward pass of the model
            out = self.model(data)
            loss = (1 / self.num_nodes) * self.training_loss_function(out, target)

            # Compute the gradients
            loss.backward()

            # Return sample gradient
            grads = [grads[k] + p.grad.data.detach().to(self.device) for k, p in enumerate(self.model.parameters())]

        return grads

    def get_grad_diff(self, current_weights, previous_weights):
        '''Get a local gradient'''

        # Update parameters
        self.replace_weights.step(current_weights, self.device)

        # Set model to training mode
        self.model.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Convert data to CUDA if possible
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Zero out gradients
            self.replace_weights.zero_grad()
            # Forward pass of the model
            out1 = self.model(data)
            loss1 = (1 / self.num_nodes) * self.training_loss_function(out1, target)
            # Compute the gradients
            loss1.backward()
            # Save gradients
            curr_grad = [p.grad.data.detach().clone().to(self.device) for p in self.model.parameters()]

            # Do it again
            self.replace_weights.step(previous_weights, self.device)
            self.replace_weights.zero_grad()
            out2 = self.model(data)
            loss2 = (1 / self.num_nodes) * self.training_loss_function(out2, target)
            loss2.backward()
            prev_grad = [p.grad.data.detach().clone().to(self.device) for p in self.model.parameters()]

            return [curr_grad[i] - prev_grad[i] for i in range(len(curr_grad))]

    def compute_optimality_criteria(self, avg_weights, local_weights, training_data_full_sample, pre_comm_weights=None):
        '''
        Compute the relevant metrics for this problem

        :param avg_weights: LIST of average weights
        :param local_weights: LIST of local weights
        :param training_data_full_sample: data loader with full gradient size
        :return:
        '''

        # Compute consensus for this agent
        local_violation = sum([numpy.linalg.norm(
            local_weights[i].cpu().numpy().flatten() - avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(local_weights))])

        # Compute the norm of the iterate to save in case consensus is large
        avg_weight_norm = sum([numpy.linalg.norm(avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(avg_weights))])

        # Compute the gradient at the average solution on this dataset:
        # 1. Replace the model params
        # 2. Forward pass, backward pass to have gradient
        # 3. Compute the stationarity violation
        # 4. MUST SCALE: total number of samples is (N * num_local) samples. Since `get_average_param` divides by N
        # the loss function here must be scaled only by (1 / num_local)
        loss_function = torch.nn.NLLLoss(reduction='sum')
        coef = 1. / (len(training_data_full_sample.dataset) // size)

        self.replace_weights.step(avg_weights, self.device)
        self.model.train()
        grads = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        for batch_idx, (data, target) in enumerate(training_data_full_sample):
            # Print errors (just in case) and zero out the gradient
            torch.autograd.set_detect_anomaly(True)
            self.replace_weights.zero_grad()
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Forward and backward pass of the model; scale by (1 / N) to line up with average
            out = self.model(data)
            loss = coef * loss_function(out, target)
            loss.backward()

            # Save gradients
            grads = [grads[ind] + p.grad.data.detach().clone().to(self.device) for ind, p in
                     enumerate(self.model.parameters())]

        # Get the average gradient by doing all_reduce and then compute the stationarity violation at the average point
        avg_grads = self.get_average_param(grads)
        stationarity1 = self.regularizer.forward([avg_weights[pa] - avg_grads[pa] for pa in range(self.num_params)],
                                                 self.l1)
        stationarity = numpy.concatenate([avg_weights[pa].detach().cpu().numpy().flatten()
                                          - stationarity1[pa].detach().cpu().numpy().flatten() for pa in
                                          range(self.num_params)])
        global_norm = numpy.linalg.norm(stationarity, ord=2) ** 2

        # Before sending, also get then number of non-zeros for this agent and this average
        if pre_comm_weights is None:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(local_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)
        else:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(pre_comm_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violation, local_nnz_ratio])
        recv_array = numpy.empty(shape=array_to_send.shape)
        comm.Barrier()
        comm.Allreduce(array_to_send, recv_array)
        comm.Barrier()

        # return consensus, gradient, total optimality, iterate history,
        # local number non-zeros, number nonzeros at everate, and average number of nonzeros
        return recv_array[0], global_norm, recv_array[0] + global_norm, avg_weight_norm, \
               nnz_at_average, (1 / size) * recv_array[1]

    def test(self, weights, testing_data, mode='global'):
        '''Test the data using the average weights'''

        self.replace_weights.zero_grad()
        self.replace_weights.step(weights, self.device)
        self.model.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():
            # Loop over testing data
            for data, target in testing_data:
                # Use CUDA
                data, target = data.to(self.device).float(), target.to(self.device).long()

                # Evaluate the model on the testing data
                output = self.model(data)
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / size)

        # We have two modes of reporting data:
        # 1. We use the AVG weights on all of the data
        # 2. We use the local weights on the local data and THEN compute average
        if mode == 'global':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            array_to_send = numpy.array([correct, num_test_points, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[2] / recv_array[1]
            testing_accuracy = 100 * recv_array[0] / recv_array[1]

        # Compue local information and then average
        elif mode == 'local':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            correct /= num_test_points
            test_loss /= num_test_points
            array_to_send = numpy.array([correct, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[1] / size
            testing_accuracy = 100 * recv_array[0] / size
        else:
            sys.exit(f"[ERROR] _ {mode} _ is not a vaild report metric; choose from \'local\' or \'global\' [ERROR]")

        return test_loss, testing_accuracy


if __name__=='__main__':

    # Parse user input
    parser = argparse.ArgumentParser(description='Testing SPPDM on problems from paper.')

    parser.add_argument('--updates', type=int, default=5000, help='Total number of communication rounds.')
    parser.add_argument('--mom', type=str, default='nesterov', choices=['nesterov', 'none', 'constant'], help='Momentum type.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Local learning rate.')
    parser.add_argument('--beta', type=float, default=0.9, help='Local learning rate.')
    parser.add_argument('--c', type=float, default=1, help='Local learning rate.')
    parser.add_argument('--gamma', type=float, default=3, help='Local learning rate.')
    parser.add_argument('--kappa', type=float, default=0.1, help='Local learning rate.')
    parser.add_argument('--l1', type=float, default=0.0, help='L-1 Regularizer.')
    parser.add_argument('--mini_batch', type=int, default=64, help='Mini-batch size.')
    parser.add_argument('--init_batch', type=int, default=1, help='Initial batch size.')
    parser.add_argument('--comm_pattern', type=str, default='ring', choices=['ring', 'random', 'complete', 'ladder'],
                        help='Communication pattern.')
    parser.add_argument('--data', type=str, default='a9a', choices=['a9a', 'mnist', 'miniboone'],
                        help='Dataset.')
    parser.add_argument('--trial', type=int, default=1, help='Which starting variables to use.')
    parser.add_argument('--report', type=int, default=100, help='How often to report criteria.')

    # Create callable argument
    args = parser.parse_args()

    ###########################
    # a9a data
    if args.data == 'a9a':
        # Subset data to local agent
        num_samples = 32561 // size
        train_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
             range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 16281 // size
        test_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=False),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    ###########################

    ###########################
    # miniboone data
    elif args.data == 'miniboone':
        # Subset data to local agent
        num_samples = 100000 // size
        train_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in
                 range(int(rank * num_samples),
                       int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 30064 // size
        test_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=False),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    ###########################

    ###########################
    # MNIST data
    else:
        # Create transform for data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

        # Subset data to local agent
        num_samples = 60000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                       transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                       transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
             range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 10000 // size
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    ###########################

    # Load communication matrix and initial weights
    mixing_matrix = torch.tensor(numpy.load(f'mixing_matrices/{args.comm_pattern}_{size}.dat', allow_pickle=True))
    arch_size = 4 if args.data in ['a9a', 'miniboone'] else 8
    init_weights = [numpy.load(os.path.join(os.getcwd(), f'init_weights/{args.data}/trial{args.trial}/rank{rank}/layer{l}.dat'),
                       allow_pickle=True) for l in range(arch_size)]

    # Print training information
    if rank == 0:
        opening_statement = f' SPPDM on {args.data} '
        print(f"\n{'#' * 75}")
        print('\n' + opening_statement.center(75, ' '))
        print(
            f'[GRAPH INFO] {size} agents | connectivity = {args.comm_pattern} | rho = {torch.sort(torch.eig(mixing_matrix)[0][:, 0])[0][size - 2].item()}')
        print(f'[TRAINING INFO] mini-batch = {args.mini_batch} | learning rate = {args.alpha}\n')
        print(f"{'#' * 75}\n")

    # Barrier before training
    comm.Barrier()

    # Declare and train!
    algo_params = {'alpha': args.alpha, 'gamma': args.gamma, 'c': args.c, 'kappa': args.kappa,
                   'beta': args.beta, 'mini_batch': args.mini_batch, 'report': args.report, 'momentum': args.mom,
                   'l1': args.l1}
    solver = SPPDM(algo_params, mixing_matrix, train_loader, init_weights)
    algo_time = solver.solve(args.updates, optimality_loader, test_loader)

    # Save the information
    method = 'sppdm'

    # Make directory for both the dataset and the method and the model
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/'))
    except:
        # Main storage already exists already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{args.data}'))
    except:
        # Method already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{args.data}/{method}'))
    except:
        # Model already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{args.data}/{method}/trial{args.trial}'))
    except:
        # Trial already exists
        pass
    try:
        os.mkdir(
            os.path.join(os.getcwd(), f'results/{args.data}/{method}/trial{args.trial}/{args.comm_pattern}{size}'))
    except:
        # Graph and size already exists
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(),
                              f'results/{args.data}/{method}/trial{args.trial}/{args.comm_pattern}{size}/{args.mini_batch}'))
    except:
        # Mini-batch already exists
        pass

    # Save path
    path = os.path.join(os.getcwd(),
                        f'results/{args.data}/{method}/trial{args.trial}/{args.comm_pattern}{size}/{args.mini_batch}')

    # Save information via numpy
    if rank == 0:
        numpy.savetxt(
            f'{path}/test_loss_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.testing_loss, fmt='%.16f')
        numpy.savetxt(
            f'{path}/test_acc_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.testing_accuracy, fmt='%.16f')
        numpy.savetxt(
            f'{path}/train_loss_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.training_loss, fmt='%.16f')
        numpy.savetxt(
            f'{path}/train_acc_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.training_accuracy, fmt='%.16f')
        numpy.savetxt(
            f'{path}/test_loss_local_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.testing_loss_local, fmt='%.16f')
        numpy.savetxt(
            f'{path}/test_acc_local_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.testing_accuracy_local, fmt='%.16f')
        numpy.savetxt(
            f'{path}/train_loss_local_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.training_loss_local, fmt='%.16f')
        numpy.savetxt(
            f'{path}/train_acc_local_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.training_accuracy_local, fmt='%.16f')
        numpy.savetxt(
            f'{path}/total_opt_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.total_optimality, fmt='%.16f')
        numpy.savetxt(
            f'{path}/consensus_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.consensus_violation, fmt='%.16f')
        numpy.savetxt(
            f'{path}/norm_hist_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.norm_hist, fmt='%.16f')
        numpy.savetxt(
            f'{path}/iterate_hist_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.iterate_norm_hist, fmt='%.16f')
        numpy.savetxt(
            f'{path}/total_time_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.total_time, fmt='%.16f')
        numpy.savetxt(
            f'{path}/comm_time_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.communication_time, fmt='%.16f')
        numpy.savetxt(
            f'{path}/comp_time_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.compute_time, fmt='%.16f')
        numpy.savetxt(
            f'{path}/nnz_at_avg_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.nnz_at_avg, fmt='%.16f')
        numpy.savetxt(
            f'{path}/avg_nnz_mom{args.mom}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_c{args.c}_kappa{args.kappa}_l1{args.l1}.txt',
            solver.avg_nnz, fmt='%.16f')

    # Barrier at end so all agents stop this script before moving on
    comm.Barrier()
