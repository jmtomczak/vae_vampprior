from __future__ import print_function
import argparse

import torch
import torch.optim as optim

import os

from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

def define_args():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE+VampPrior')
    # arguments for optimization
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='N',
                        help='number of epochs for early stopping')

    parser.add_argument('--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warmu-up')

    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    # random seed
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # model: latent size, input_size
    parser.add_argument('--z1_size', type=int, default=40, metavar='N',
                        help='latent size')
    parser.add_argument('--z2_size', type=int, default=40, metavar='N',
                        help='latent size')
    parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='N',
                        help='input size')

    parser.add_argument('--activation', type=str, default=None, metavar='N',
                        help='activation function')

    parser.add_argument('--number_components', type=int, default=500, metavar='N',
                        help='number of pseudo-inputs')

    # model name
    parser.add_argument('--model_name', type=str, default='vae', metavar='N',
                        help='model name: vae, vae_vampprior, vae_vampprior_2level')
    # dataset
    parser.add_argument('--dataset_name', type=str, default='static_mnist', metavar='N',
                        help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes')

    parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                        help='allow dynamic binarization')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return args, kwargs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    model_name = args.model_name + '(K_' + str(args.number_components) + ')' + '_wu(' + str(args.warmup) + ')' + '_z1_' + str(args.z1_size) + '_z2_' + str(args.z2_size)

    print(args)

    with open('vae_experiment_log.txt', 'a') as f:
        print(args, file=f)

    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + model_name + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')
    # loading data
    train_loader, val_loader, test_loader = load_dataset(args)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'vae_vampprior':
        from models.VAE_VampPrior import VAE
    elif args.model_name == 'vae_vampprior_2level':
        from models.VAE_VampPrior_2level import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print('perform experiment')
    from utils.perform_experiment import experiment_vae
    experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name = args.model_name)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('vae_experiment_log.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    args, kwargs = define_args()
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #