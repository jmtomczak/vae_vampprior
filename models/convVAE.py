from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear, \
    Conv2d, GatedConv2d, MaskedConv2d

from Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.q_z_layers = nn.ModuleList()

        # conv 0
        self.q_z_layers.append(GatedConv2d(1, 32, 5, 1, 3))
        # conv 1
        self.q_z_layers.append(GatedConv2d(32, 32, 5, 2, 1))
        # conv 2
        self.q_z_layers.append(GatedConv2d(32, 64, 5, 1, 3))
        # conv 3
        self.q_z_layers.append(GatedConv2d(64, 64, 5, 2, 1))
        # conv 7
        self.q_z_layers.append(GatedConv2d(64, 6, 3, 1, 1))

        # linear layers
        self.q_z_mean = nn.Linear(210, self.args.z1_size)
        self.q_z_logvar = NonLinear(210, self.args.z1_size, activation=nn.Hardtanh(min_value=-6., max_value=2.))

        # decoder: p(x | z)
        self.p_x_layers = nn.ModuleList()

        self.p_x_layers.append( GatedDense(self.args.z1_size, np.prod(self.args.input_size)) )

        # PixelCNN
        act = nn.ReLU()
        self.pixelcnn = nn.Sequential(MaskedConv2d('A', 2, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act,
            MaskedConv2d('B', 64, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), act )

        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(64, 1, 1, 1, 0, activation=nn.Hardtanh(min_value=-5., max_value=0.))

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        # elif self.args.input_type == 'continuous':
        #     RE = log_Normal_diag(x, x_mean, x_logvar, dim=1) + 0.5*self.args.input_size[0]*self.args.input_size[1]*self.args.input_size[2]*math.log(2.*math.pi)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
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

    def calculate_lower_bound(self, X_full, MB=100):
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

    # ADDITIONAL METHODS
    def generate_x(self, N=25):
        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N].view(-1,self.args.input_size[0],self.args.input_size[1],self.args.input_size[2])
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        # sample from PixelCNN
        x_zeros = torch.zeros((z_sample_rand.size(0), self.args.input_size[0], self.args.input_size[1], self.args.input_size[2]))
        if self.args.cuda:
            x_zeros = x_zeros.cuda()

        for i in range(self.args.input_size[1]):
            for j in range(self.args.input_size[2]):
                samples_rand, _ = self.p_x(Variable(x_zeros, volatile=True), z_sample_rand)
                samples_rand = samples_rand.view(samples_rand.size(0), self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

                probs = samples_rand[:, :, i, j].data
                x_zeros[:, :, i, j] = torch.bernoulli(probs).float()

        return samples_rand

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        for i in range(len(self.q_z_layers)):
            x = self.q_z_layers[i](x)

        h = x.view(-1,210)

        z_q_mean = self.q_z_mean(h)
        z_q_logvar = self.q_z_logvar(h)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, x, z):
        # process z
        for j in range(len(self.p_x_layers)):
            z = self.p_x_layers[j](z)
        z = z.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

        # concatenate x and z
        h = torch.cat((x,z), 1)

        # pixelcnn part of the decoder
        h_pixelcnn = self.pixelcnn(h)

        x_mean = self.p_x_mean(h_pixelcnn).view(-1,np.prod(self.args.input_size))
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_logvar = self.p_x_logvar(h_pixelcnn).view(-1,np.prod(self.args.input_size))
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            MB = z.size(0)
            C = self.args.number_components
            M = z.size(1)

            # calculate params
            X = self.means(self.idle_input).view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

            # calculate params for given data
            z_p_mean, z_p_logvar = self.q_z(X)  # C x M

            # expand z
            z_expand = z.unsqueeze(1).expand(MB, C, M)
            means = z_p_mean.unsqueeze(0).expand(MB, C, M)
            logvars = z_p_logvar.unsqueeze(0).expand(MB, C, M)

            a = log_Normal_diag(z_expand, means, logvars, dim=2).squeeze(2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB x 1
            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.expand(MB, C)), 1))  # MB x 1

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(x, z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar