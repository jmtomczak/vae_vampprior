from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np

import math

from utils.distributions import log_Normal_diag, log_Normal_standard, log_Bernoulli
from utils.visual_evaluation import plot_reconstruction, plot_scatter, plot_real, plot_generation, plot_manifold, plot_images

import time
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        x = data
        # forward pass
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = model.forward(x)
        # loss function
        # RE
        RE = log_Bernoulli(x, x_mean)

        # KL
        log_p_z = log_Normal_standard(z_q, dim=1)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = - torch.sum(log_p_z - log_q_z)

        evaluate_loss += ((-RE + KL) / data.size(0)).data[0]
        evaluate_re += (-RE / data.size(0)).data[0]
        evaluate_kl += (KL / data.size(0)).data[0]

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            plot_reconstruction(args, x_mean, epoch, dir, size_x=3, size_y=3)
            if epoch == 1:
                # VISUALIZATION: plot real images
                plot_real(args, data[:26], dir + 'reconstruction/', size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        test_data = Variable(data_loader.dataset.data_tensor)
        test_target = Variable(data_loader.dataset.target_tensor)
        full_data = Variable(train_loader.dataset.data_tensor)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        full_data = torch.bernoulli(full_data)

        # VISUALIZATION: plot real images
        plot_real(args, test_data, dir, size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        z_mean_recon, z_logvar_recon = model.q_z(test_data)
        z_recon = model.reparameterize(z_mean_recon, z_logvar_recon)
        samples, _ = model.p_x(z_recon)

        plot_reconstruction(args, samples, epoch, dir, size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        z_sample_rand = Variable(torch.normal( torch.from_numpy( np.zeros((25,args.z1_size)) ).float(), 1. ) )
        if args.cuda:
            z_sample_rand = z_sample_rand.cuda()

        samples_rand, _ = model.p_x(z_sample_rand)

        plot_generation(args, samples_rand, dir, size_x=5, size_y=5)

        if args.z1_size == 2:
            # VISUALIZATION: plot low-dimensional manifold
            plot_manifold(model, args, dir)

            # VISUALIZATION: plot scatter-plot
            plot_scatter(model, test_data, test_target, dir)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl

# ======================================================================================================================
def evaluate_vae_vampprior(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        x = data
        # forward pass
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = model.forward(x)
        # loss function
        # RE
        RE = log_Bernoulli(x, x_mean)

        # KL
        log_p_z = model.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = - torch.sum(log_p_z - log_q_z)

        evaluate_loss += ((-RE + KL) / data.size(0)).data[0]
        evaluate_re += (-RE / data.size(0)).data[0]
        evaluate_kl += (KL / data.size(0)).data[0]

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            plot_reconstruction(args, x_mean, epoch, dir, size_x=3, size_y=3)
            if epoch == 1:
                # VISUALIZATION: plot real images
                plot_real(args, data[:26], dir + 'reconstruction/', size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        test_data = Variable(data_loader.dataset.data_tensor)
        test_target = Variable(data_loader.dataset.target_tensor)
        full_data = Variable(train_loader.dataset.data_tensor)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        full_data = torch.bernoulli(full_data)

        # VISUALIZATION: plot real images
        plot_real(args, test_data, dir, size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        z_mean_recon, z_logvar_recon = model.q_z(test_data)
        z_recon = model.reparameterize(z_mean_recon, z_logvar_recon)
        samples, _ = model.p_x(z_recon)

        plot_reconstruction(args, samples, epoch, dir, size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        means = model.means(model.idle_input)[0:25]
        z_sample_gen_mean, z_sample_gen_logvar = model.q_z(means)

        # calulate sample z ~ N(mu, 1)
        z_sample_rand = model.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
        samples_rand, _ = model.p_x(z_sample_rand)

        plot_generation(args, samples_rand, dir, size_x=5, size_y=5)

        if args.z1_size == 2:
            # VISUALIZATION: plot low-dimensional manifold
            plot_manifold(model, args, dir)

            # VISUALIZATION: plot scatter-plot
            plot_scatter(model, test_data, test_target, dir)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0.  # model.calculate_likelihood(full_data, dir, mode='train')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl

# ======================================================================================================================
def evaluate_vae_vampprior_2level(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        x = data
        # forward pass
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model.forward(
            x)
        # loss function
        # RE
        RE = log_Bernoulli(x, x_mean)

        # KL
        log_p_z2 = model.log_p_z2(z2_q)
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = - torch.sum(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        evaluate_loss += ((-RE + KL) / data.size(0)).data[0]
        evaluate_re += (-RE / data.size(0)).data[0]
        evaluate_kl += (KL / data.size(0)).data[0]

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            plot_reconstruction(args, x_mean, epoch, dir, size_x=3, size_y=3)
            if epoch == 1:
                # VISUALIZATION: plot real images
                plot_real(args, data[:26], dir + 'reconstruction/', size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        test_data = Variable(data_loader.dataset.data_tensor)
        test_target = Variable(data_loader.dataset.target_tensor)
        full_data = Variable(train_loader.dataset.data_tensor)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        full_data = torch.bernoulli(full_data)

        # VISUALIZATION: plot real images
        plot_real(args, test_data, dir, size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        z2_mean_recon, z2_logvar_recon = model.q_z2(test_data)
        z2_recon = model.reparameterize(z2_mean_recon, z2_logvar_recon)
        z1_mean_recon, z1_logvar_recon = model.q_z1(test_data, z2_recon)
        z1_recon = model.reparameterize(z1_mean_recon, z1_logvar_recon)
        samples, _ = model.p_x(z1_recon, z2_recon)

        plot_reconstruction(args, samples, epoch, dir, size_x=5, size_y=5)


        # VISUALIZATION: plot generations
        means = model.means(model.idle_input)[0:25]

        z2_sample_gen_mean, z2_sample_gen_logvar = model.q_z2(means)
        z2_sample_rand = model.reparameterize(z2_sample_gen_mean, z2_sample_gen_logvar)

        z1_mean_rand, z1_logvar_rand = model.p_z1(z2_sample_rand)
        z1_sample_rand = model.reparameterize(z1_mean_rand, z1_logvar_rand)
        samples_rand, _ = model.p_x(z1_sample_rand, z2_sample_rand)

        plot_generation(args, samples_rand, dir, size_x=5, size_y=5)

        if args.z1_size == 2 and args.z2_size == 2:
            # VISUALIZATION: plot low-dimensional manifold
            from utils.visual_evaluation import plot_manifold2
            plot_manifold2(model, args, dir)

            # VISUALIZATION: plot scatter-plot
            from utils.visual_evaluation import plot_scatter2
            z2_mean_recon, z2_logvar_recon = model.q_z2(test_data)
            z2_recon = model.reparameterize(z2_mean_recon, z2_logvar_recon)

            plot_scatter2(model, z2_recon, test_target, dir, name='scatter2D_z2.png')

            z1_mean_recon, z1_logvar_recon = model.q_z1(test_data, z2_recon)
            z1_recon = model.reparameterize(z1_mean_recon, z1_logvar_recon)
            plot_scatter2(model, z1_recon, test_target, dir, name='scatter2D_z1_1.png')

            z1_mean_recon, z1_logvar_recon = model.q_z1(test_data, z2_recon)
            z1_recon = model.reparameterize(z1_mean_recon, z1_logvar_recon)
            plot_scatter2(model, z1_recon, test_target, dir, name='scatter2D_z1_2.png')

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0.  # model.calculate_likelihood(full_data, dir, mode='train')
        t_ll_e = time.time()
        print('Log_likelihood time: {:.2f}'.format(t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl
