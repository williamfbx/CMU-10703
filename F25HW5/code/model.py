import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging

log = logging.getLogger("root")


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(
            -3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )
        self.min_logvar = torch.tensor(
            -7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )

        # Create or load networks
        self.networks = nn.ModuleList(
            [self.create_network(n) for n in range(self.num_nets)]
        ).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0 : self.state_dim]
        raw_v = output[:, self.state_dim :]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here

        # negative log likelihood of the observed next states under the predicted mean and variance from the network,
        # conditioned on the observed current states and actions

        nll = 0.5 * (logvar + (targ - mean)**2 / torch.exp(logvar)).mean()

        return nll

    def create_network(self, n):
        layer_sizes = [
            self.state_dim + self.action_dim,
            HIDDEN1_UNITS,
            HIDDEN2_UNITS,
            HIDDEN3_UNITS,
        ]
        layers = reduce(
            operator.add,
            [
                [nn.Linear(a, b), nn.ReLU()]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here
        avg_loss = []

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

        for _ in range(num_train_itrs):
            this_itr_loss = []
            for n in range(self.num_nets):
                # uniformly sample (with replacement) minibatch of size B from D
                minibatch_ind = np.random.randint(0, len(inputs), size=batch_size)
                minibatch = inputs[minibatch_ind]
                mean, logvar = self.get_output(self.networks[n](minibatch))

                # take a gradient step of the loss for sampled minibatch
                self.opt.zero_grad()
                loss = self.get_loss(targets[minibatch_ind], mean, logvar)
                loss.backward()
                self.opt.step()
                this_itr_loss.append(loss.item())

            avg_loss.append(np.mean(this_itr_loss))

        return avg_loss
