from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np

import math

from utils.distributions import log_Normal_diag, log_Normal_standard, log_Bernoulli
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_vae(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* (epoch-1) / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
	if args.dynamic_binarization:
            x = torch.bernoulli(data)
	else:
	    x = data
        # reset gradients
        optimizer.zero_grad()
        # forward pass
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = model.forward(x)
        # loss function
        RE = log_Bernoulli(data, x_mean, average=False)
        # KL
        log_p_z = log_Normal_standard(z_q, dim=1)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = beta * (- torch.sum(log_p_z - log_q_z) )

        loss = (-RE + KL) / data.size(0)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data[0]
        train_re += (-RE / data.size(0)).data[0]
        train_kl += (KL / data.size(0)).data[0]

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl

# ======================================================================================================================
def train_vae_vampprior(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* (epoch-1) / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
	else:
	    x = data
        # reset gradients
        optimizer.zero_grad()
        # forward pass
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = model.forward(x)
        # loss function
        # RE
        RE = log_Bernoulli(x, x_mean)

        # KL
        log_p_z = model.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = beta * (- torch.sum(log_p_z - log_q_z) )

        loss = (-RE + KL) / data.size(0)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data[0]
        train_re += (-RE / data.size(0)).data[0]
        train_kl += (KL / data.size(0)).data[0]

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl

# ======================================================================================================================
def train_vae_vampprior_2level(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* (epoch-1) / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
	else:
	    x = data
        # reset gradients
        optimizer.zero_grad()
        # forward pass
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model.forward(x)
        # loss function
        # RE
        RE = log_Bernoulli(x, x_mean)

        # KL
        log_p_z2 = model.log_p_z2(z2_q)
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = beta * (- torch.sum(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2) )

        loss = (-RE + KL) / data.size(0)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data[0]
        train_re += (-RE / data.size(0)).data[0]
        train_kl += (KL / data.size(0)).data[0]

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl
