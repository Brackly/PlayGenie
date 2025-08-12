import torch
from torch._C.cpp import nn

def kl_loss(mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

def generative_loss(x,x_hat,mu, log_var)->torch.Tensor:
    _loss = torch.nn.functional.mse_loss(x,x_hat)
    return _loss + kl_loss(mu, log_var)


def discriminative_loss(x,x_hat,mu, log_var)->torch.Tensor:
    _loss = torch.nn.functional.cross_entropy(x,x_hat)
    return _loss + kl_loss(mu, log_var)