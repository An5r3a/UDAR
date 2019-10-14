import numpy as np
import torch
import torch.nn.functional as F

def vae_loss(q_mu, q_logvar, output, target):
    return mse_loss(output, target), kld_gauss(q_mu, q_logvar)


def mse_loss(output, target, avg_batch=True):
    """
    Reconstruction loss
    """
    output = F.mse_loss(output, target, reduction='none')
    output = torch.sum(output)  # sum over all TF units
    if avg_batch:
        output = torch.mean(output, dim=0)
    # return F.mse_loss(output, target, reduction=reduce)  # careful about the scaling
    return output


def kld_gauss(q_mu, q_logvar, mu=None, logvar=None, avg_batch=True):
    """
    KL divergence between two diagonal Gaussians
    in standard VAEs, the prior p(z) is a standard Gaussian.
    :param q_mu: posterior mean
    :param q_logvar: posterior log-variance
    :param mu: prior mean
    :param logvar: prior log-variance
    """
    # set prior to a standard Gaussian
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    output = torch.sum(1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar),
                        dim=1)
    output *= -0.5
    if avg_batch:
        output = torch.mean(output, dim=0)
    return output