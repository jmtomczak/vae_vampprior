from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

min_epsilon = 1e-5
max_epsilon = 1.-1e-5
#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )

def logisticCDF(x, u, s):
    return 1. / ( 1. + torch.exp( -(x-u) / s ) )

def sigmoid(x):
    return 1. / ( 1. + torch.exp( -x ) )

def log_Logistic_256(x, mean, log_s, average=False, dim=None):
    binsize = 1. / 256.
    scale = torch.exp(log_s)
    # make sure image fit proper values
    x = torch.floor(x/binsize) * binsize
    # calculate normalized values for a bin
    x_plus = (x + binsize - mean) / scale
    x_minus = (x - mean) / scale
    # calculate logistic CDF for a bin
    cdf_plus = sigmoid(x_plus)
    cdf_minus = sigmoid(x_minus)
    # calculate final log-likelihood for an image
    log_logistic_256 = - torch.log( cdf_plus - cdf_minus + 1.e-7 )

    if average:
        return torch.mean( log_logistic_256, dim )
    else:
        return torch.sum( log_logistic_256, dim )