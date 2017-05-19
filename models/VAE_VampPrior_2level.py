from __future__ import print_function

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard
from utils.visual_evaluation import plot_histogram

import numpy as np
import math

from scipy.misc import logsumexp

def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g

#=======================================================================================================================
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.args = args

        # Model:
        # decoder: p(x | z1, z2)
        self.p_x_layers_pre = nn.ModuleList()
        self.p_x_layers_gate = nn.ModuleList()

        self.p_x_layers_pre.append( nn.Linear(self.args.z1_size, 300) )
        self.p_x_layers_gate.append( nn.Linear(self.args.z1_size, 300) )

        self.p_x_layers_pre.append( nn.Linear(self.args.z2_size, 300) )
        self.p_x_layers_gate.append( nn.Linear(self.args.z2_size, 300) )

        self.p_x_layers_pre.append( nn.Linear(2*300, 300) )
        self.p_x_layers_gate.append( nn.Linear(2*300, 300) )

        self.p_x_mean = nn.Linear(300, np.prod(self.args.input_size))

        # p(z1|z2)
        self.p_z1_layers_pre = nn.ModuleList()
        self.p_z1_layers_gate = nn.ModuleList()

        self.p_z1_layers_pre.append( nn.Linear(self.args.z2_size, 300) )
        self.p_z1_layers_gate.append( nn.Linear(self.args.z2_size, 300) )

        self.p_z1_layers_pre.append( nn.Linear(300, 300) )
        self.p_z1_layers_gate.append( nn.Linear(300, 300) )

        self.p_z1_mean = nn.Linear(300, self.args.z1_size)
        self.p_z1_logvar = nn.Linear(300, self.args.z1_size)

        # p(z2) = 1/K sum_n q(z2 | x_k)
        # mixture of Gaussians parameters
        self.means = nn.Linear(self.args.number_components, np.prod(self.args.input_size), bias=False)
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components))
        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()

        # Variational:
        # q(z1|x,z2)
        self.q_z1_layers_pre = nn.ModuleList()
        self.q_z1_layers_gate = nn.ModuleList()

        self.q_z1_layers_pre.append( nn.Linear(np.prod(self.args.input_size), 300) )
        self.q_z1_layers_gate.append( nn.Linear(np.prod(self.args.input_size), 300) )

        self.q_z1_layers_pre.append( nn.Linear(np.prod(self.args.z2_size), 300) )
        self.q_z1_layers_gate.append( nn.Linear(np.prod(self.args.z2_size), 300) )

        self.q_z1_layers_pre.append( nn.Linear(2 * 300, 300) )
        self.q_z1_layers_gate.append( nn.Linear(2 * 300, 300) )

        self.q_z1_mean = nn.Linear(300, self.args.z1_size)
        self.q_z1_logvar = nn.Linear(300, self.args.z1_size)

        # q(z2 | x)
        self.q_z2_layers_pre = nn.ModuleList()
        self.q_z2_layers_gate = nn.ModuleList()

        self.q_z2_layers_pre.append( nn.Linear(np.prod(self.args.input_size), 300) )
        self.q_z2_layers_gate.append( nn.Linear(np.prod(self.args.input_size), 300) )

        self.q_z2_layers_pre.append( nn.Linear(300, 300) )
        self.q_z2_layers_gate.append( nn.Linear(300, 300) )

        self.q_z2_mean = nn.Linear(300, self.args.z2_size)
        self.q_z2_logvar = nn.Linear(300, self.args.z2_size)

        self.sigmoid = nn.Sigmoid()

        self.Gate = Gate()

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m)

    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_likelihood(self, X, dir, mode='test', S=5000):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        MB = 100

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, R):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                # pass through VAE
                x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

                # RE
                RE = log_Bernoulli(x, x_mean, dim=1)

                # KL
                log_p_z2 = self.log_p_z2(z2_q)
                log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
                log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
                log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
                KL = - (log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

                a_tmp = (RE - KL)

                a.append( a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(S))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        MB = 100

        for i in range(X_full.size(0) / MB):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            # pass through VAE
            x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

            # RE
            RE = log_Bernoulli( x, x_mean )

            # KL
            log_p_z2 = self.log_p_z2(z2_q)
            log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
            log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
            log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
            KL = - torch.sum(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]

            # CALCULATE LOWER-BOUND: RE + KL - ln(N)
            lower_bound += (-RE + KL).cpu().data[0]

        lower_bound = lower_bound / X_full.size(0)

        return lower_bound

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z1(self, x, z2):
        h0_pre_x = self.q_z1_layers_pre[0](x)
        h0_gate_x = self.sigmoid( self.q_z1_layers_gate[0](x) )
        h0_x = self.Gate( h0_pre_x, h0_gate_x )

        h0_pre_z2 = self.q_z1_layers_pre[1](z2)
        h0_gate_z2 = self.sigmoid( self.q_z1_layers_gate[1](z2) )
        h0_z2 = self.Gate( h0_pre_z2, h0_gate_z2 )

        h0 = torch.cat( (h0_x, h0_z2), 1)

        h1_pre = self.q_z1_layers_pre[2](h0)
        h1_gate = self.sigmoid( self.q_z1_layers_gate[2](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z1_q_mean = self.q_z1_mean(h1)
        z1_q_logvar = self.q_z1_logvar(h1)
        return z1_q_mean, z1_q_logvar

    def q_z2(self, x):
        h0_pre = self.q_z2_layers_pre[0](x)
        h0_gate = self.sigmoid( self.q_z2_layers_gate[0](x) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.q_z2_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.q_z2_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z2_q_mean = self.q_z2_mean(h1)
        z2_q_logvar = self.q_z2_logvar(h1)
        return z2_q_mean, z2_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z1, z2):
        h0_pre_z1 = self.p_x_layers_pre[0](z1)
        h0_gate_z1 = self.sigmoid( self.p_x_layers_gate[0](z1) )
        h0_z1 = self.Gate( h0_pre_z1, h0_gate_z1 )

        h0_pre_z2 = self.p_x_layers_pre[1](z2)
        h0_gate_z2 = self.sigmoid( self.p_x_layers_gate[1](z2) )
        h0_z2 = self.Gate( h0_pre_z2, h0_gate_z2 )

        h0 = torch.cat( (h0_z1, h0_z2), 1 )

        h1_pre = self.p_x_layers_pre[2](h0)
        h1_gate = self.sigmoid( self.p_x_layers_gate[2](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        x_mean = self.sigmoid( self.p_x_mean(h1) )
        x_logvar = 0.
        return x_mean, x_logvar

    def p_z1(self, z2):
        h0_pre = self.p_z1_layers_pre[0](z2)
        h0_gate = self.sigmoid( self.p_z1_layers_gate[0](z2) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.p_z1_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.p_z1_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z1_p_mean = self.p_z1_mean(h1)
        z1_p_logvar = self.p_z1_logvar(h1)
        return z1_p_mean, z1_p_logvar

    def log_p_z2(self, z2):
        # z1 - MB x M
        # X - N x D
        MB = z2.size(0)
        C = self.args.number_components
        M = z2.size(1)

        # calculate params for given data
        X = self.means(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.q_z2(X) #C x M

        # expand z
        z_expand = z2.unsqueeze(1).expand(MB, C, M)
        means = z_p_mean.unsqueeze(0).expand(MB, C, M)
        logvars = z_p_logvar.unsqueeze(0).expand(MB, C, M)

        a = log_Normal_diag( z_expand, means, logvars, dim=2 ).squeeze(2) - math.log(C) # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1
        # calculte log-sum-exp
        log_p_z = a_max + torch.log(torch.sum(torch.exp(a - a_max.expand(MB, C)), 1))  # MB x 1

        return log_p_z

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # z1_p_mean, z1_p_logvar = p(z1|z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)

        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar