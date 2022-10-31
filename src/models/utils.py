"""Helper functions for models."""

from typing import Any
import torch
from copy import deepcopy
from torch import optim

def copy_gen_params(generator: Any):
    """
    Function to copy the parameters of the generator
    """
    params = deepcopy(list(p.data for p in generator.parameters()))
    return params

def define_optimizers(generator: Any, discriminator: Any, disc_lr: float, gen_lr: float):
    """
    Function to define the optimizers for the generator and discriminator
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param disc_lr: Learning rate for the discriminator
    :param gen_lr: Learning rate for the generator
    """
    optimizer_G = optim.Adam(generator.parameters(), lr = gen_lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr = disc_lr, betas=(0.5, 0.999))

    return optimizer_G, optimizer_D

def prepare_labels(batch_size: int, device: Any):
    """
    Function to prepare the labels for the discriminator and generator.
    """
    real_labels = torch.FloatTensor(batch_size).fill_(1).to(device)
    fake_labels = torch.FloatTensor(batch_size).fill_(0).to(device)
    match_labels = torch.LongTensor(range(batch_size)).to(device)

    return real_labels, fake_labels, match_labels

def load_params(generator: Any, params: Any):
    """
    Function to load the parameters of the generator
    """
    for p, new_p in zip(generator.parameters(), params):
        p.data.copy_(new_p)