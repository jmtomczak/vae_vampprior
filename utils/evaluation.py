from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visual_evaluation import plot_images

import time

import os
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
        # calculate loss function
        loss, RE, KL = model.calculate_loss(x, average=True)

        evaluate_loss += loss.data[0]
        evaluate_re += -RE.data[0]
        evaluate_kl += KL.data[0]

        # print N digits
        if batch_idx == 1 and mode == 'validation':
            if epoch == 1:
                if not os.path.exists(dir + 'reconstruction/'):
                    os.makedirs(dir + 'reconstruction/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:9], dir + 'reconstruction/', 'real', size_x=3, size_y=3)
            x_mean = model.reconstruct_x(x)
            plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'reconstruction/', str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        test_data = Variable(data_loader.dataset.data_tensor)
        test_target = Variable(data_loader.dataset.target_tensor)
        full_data = Variable(train_loader.dataset.data_tensor)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # print(model.means(model.idle_input))

        # VISUALIZATION: plot real images
        plot_images(args, test_data.data.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(25)

        plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            plot_images(args, pseudoinputs[0:25], dir, 'pseudoinputs', size_x=5, size_y=5)

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data, MB=args.MB)
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl