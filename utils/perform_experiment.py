from __future__ import print_function

import torch

import copy

import time

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name='vae'):
    if model_name == 'vae':
        from utils.training import train_vae as train
        from utils.evaluation import evaluate_vae as evaluate
    elif model_name == 'vae_vampprior':
        from utils.training import train_vae_vampprior as train
        from utils.evaluation import evaluate_vae_vampprior as evaluate
    elif model_name == 'vae_vampprior_2level':
        from utils.training import train_vae_vampprior_2level as train
        from utils.evaluation import evaluate_vae_vampprior_2level as evaluate
    else:
        raise Exception('Wrong name of the model!')

    best_model = model
    best_loss = 1000.
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    time_history = []

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch,
            e, args.early_stopping_epochs, best_loss
        ))

        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            best_model = model
        else:
            e += 1
            if e > args.early_stopping_epochs:
                break

        if epoch < args.warmup:
            e = 0

    # FINAL EVALUATION
    test_loss, test_re, test_kl, test_log_likelihood, train_log_likelihood, test_elbo, train_elbo = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
    ))

    with open('vae_experiment_log.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
        ), file=f)

    # SAVING
    torch.save(best_model, dir + args.model_name + '.model')
    torch.save(args, dir + args.model_name + '.config')
    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
    torch.save(test_loss, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')
