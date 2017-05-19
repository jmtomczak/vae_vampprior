from __future__ import print_function
import torch
import torch.utils.data

#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
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
    probs = torch.clamp( mean, min=1e-7, max=1.-1e-7 )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )