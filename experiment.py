from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.optimizer import AdamNormGrad

import os

import datetime

from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')
# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')
# model: latent size, input_size, so on
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=500, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='convvae_2level', metavar='MN',
                    help='model name: vae, vae_2level, convvae, convvae_2level')

parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='static_mnist', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]

    model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior + '(K_' + str(args.number_components) + ')' + '_wu(' + str(args.warmup) + ')' + '_z1_' + str(args.z1_size) + '_z2_' + str(args.z2_size)

    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + model_name + '_' + args.model_signature + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'vae_2level':
        from models.VAE_2level import VAE
    elif args.model_name == 'convvae':
        from models.convVAE import VAE
    elif args.model_name == 'convvae_2level':
        from models.convVAE_2level import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print(args)
    with open('vae_experiment_log.txt', 'a') as f:
        print(args, file=f)

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
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
