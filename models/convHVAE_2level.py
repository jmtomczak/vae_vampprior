from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear, \
    Conv2d, GatedConv2d, GatedResUnit, ResizeGatedConv2d, MaskedConv2d, ResUnitBN, ResizeConv2d, GatedResUnit, GatedConvTranspose2d

from Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        if self.args.dataset_name == 'freyfaces':
            h_size = 210
        elif self.args.dataset_name == 'cifar10':
            h_size = 384
        else:
            h_size = 294

        # encoder: q(z2 | x)
        self.q_z2_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        # linear layers
        self.q_z2_mean = NonLinear(h_size, self.args.z2_size, activation=None)
        self.q_z2_logvar = NonLinear(h_size, self.args.z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # encoder: q(z1|x,z2)
        # PROCESSING x
        self.q_z1_layers_x = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        # PROCESSING Z2
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, h_size)
        )
        # PROCESSING JOINT
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense( 2 * h_size, 300)
        )
        # linear layers
        self.q_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder p(z1|z2)
        self.p_z1_layers = nn.Sequential(
            GatedDense(self.args.z2_size, 300),
            GatedDense(300, 300)
        )
        self.p_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(300, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: p(x | z)
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, 300)
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, 300)
        )

        self.p_x_layers_joint_pre = nn.Sequential(
            GatedDense(2 * 300, np.prod(self.args.input_size))
        )

        # decoder: p(x | z)
        act = nn.ReLU(True)
        # joint
        self.p_x_layers_joint = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
        )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        # pass through VAE
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        # full loss
        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=500):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

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
                x = x_single.expand(S, x_single.size(1)).contiguous()

                a_tmp, _, _ = self.calculate_loss(x)
                a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=500):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL = self.calculate_loss(x,average=True)

            RE_all += RE.cpu().data[0]
            KL_all += KL.cpu().data[0]
            lower_bound += loss.cpu().data[0]

        lower_bound /= I

        return lower_bound

    def generate_x(self, N=25):
        # Sampling z2 from a prior
        if self.args.prior == 'standard':
            z2_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z2_sample_rand = z2_sample_rand.cuda()

        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N].view(-1, self.args.input_size[0], self.args.input_size[1],self.args.input_size[2])
            z2_sample_gen_mean, z2_sample_gen_logvar = self.q_z2(means)
            z2_sample_rand = self.reparameterize(z2_sample_gen_mean, z2_sample_gen_logvar)

        # Sampling z1 from a model
        z1_sample_mean, z1_sample_logvar = self.p_z1(z2_sample_rand)
        z1_sample_rand = self.reparameterize(z1_sample_mean, z1_sample_logvar)

        # Sampling from PixelCNN
        samples_gen, _ = self.p_x(z1_sample_rand, z2_sample_rand)

        return samples_gen

    def reconstruct_x(self, x):
        x_reconstructed, _, _, _, _, _, _, _, _, _ = self.forward(x)

        return x_reconstructed

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z2(self, x):
        # processing x
        h = self.q_z2_layers(x)
        h = h.view(x.size(0),-1)
        # predict mean and variance
        z2_q_mean = self.q_z2_mean(h)
        z2_q_logvar = self.q_z2_logvar(h)
        return z2_q_mean, z2_q_logvar

    def q_z1(self, x, z2):
        # x = x.view(x.size(0),-1)
        # processing x
        x = self.q_z1_layers_x(x)
        x = x.view(x.size(0),-1)
        # processing z2
        z2 = self.q_z1_layers_z2(z2)
        # concatenating
        h = torch.cat((x,z2),1)
        h = self.q_z1_layers_joint(h)
        # predict mean and variance
        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_z1(self, z2):
        z2 = self.p_z1_layers(z2)
        # predict mean and variance
        z1_p_mean = self.p_z1_mean(z2)
        z1_p_logvar = self.p_z1_logvar(z2)
        return z1_p_mean, z1_p_logvar

    def p_x(self, z1, z2):
        # processing z2
        z2 = self.p_x_layers_z2(z2)
        # processing z1
        z1 = self.p_x_layers_z1(z1)
        # concatenate x and z1 and z2
        h = torch.cat((z1, z2), 1)

        h = self.p_x_layers_joint_pre(h)
        h = h.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

        # joint decoder part of the decoder
        h_decoder = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h_decoder).view(-1,np.prod(self.args.input_size))
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h_decoder).view(-1,np.prod(self.args.input_size))

        return x_mean, x_logvar

    # the prior
    def log_p_z2(self, z2):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z2, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means(self.idle_input).view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

            # calculate params for given data
            z2_p_mean, z2_p_logvar = self.q_z2(X)  # C x M)

            # expand z
            z_expand = z2.unsqueeze(1)
            means = z2_p_mean.unsqueeze(0)
            logvars = z2_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB
            # calculte log-sum-exp
            log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z1,z2)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)
        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar
