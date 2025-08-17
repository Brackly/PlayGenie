import torch


def kl_loss(mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

def generative_loss(preds,targets,mu, log_var)->torch.Tensor:
    _loss = torch.nn.functional.mse_loss(preds,targets)
    return _loss + kl_loss(mu, log_var)


def discriminative_loss(preds,targets,mu, log_var)->torch.Tensor:
    _loss = torch.nn.functional.cross_entropy(preds,targets)
    return _loss + kl_loss(mu, log_var)