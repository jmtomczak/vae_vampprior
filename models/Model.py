from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.nn import normal_init, NonLinear
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        if self.args.prior == 'vampprior':
            # create pseudo-input for the VampPrior
            if self.args.input_type == 'binary' or self.args.input_type == 'gray':
                nonlinearity = nn.Sigmoid()
            elif self.args.input_type == 'continuous':
                nonlinearity = None
            self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
            # init pseudoinputs
            normal_init(self.means.linear,self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

            # create an idle input for calling pseudo-inputs
            self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
            if self.args.cuda:
                self.idle_input = self.idle_input.cuda()

    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.