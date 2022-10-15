"""Conditioning Augmentation Module"""

from typing import Any

import torch
from torch import nn


class CondAugmentation(nn.Module):
    """Conditioning Augmentation Module"""

    def __init__(self, D: int, conditioning_dim: int):
        """
        :param D: Dimension of the text embedding space [D from AttnGAN paper]
        :param conditioning_dim: Dimension of the conditioning space
        """
        super().__init__()
        self.cond_dim = conditioning_dim
        self.cond_augment = nn.Linear(D, conditioning_dim * 4, bias=True)
        self.glu = nn.GLU(dim=1)

    def encode(self, text_embedding: torch.Tensor) -> Any:
        """
        This function encodes the text embedding into the conditioning space
        :param text_embedding: Text embedding
        :return: Conditioning embedding
        """
        x_tensor = self.glu(self.cond_augment(text_embedding))
        mu_tensor = x_tensor[:, : self.cond_dim]
        logvar = x_tensor[:, self.cond_dim :]
        return mu_tensor, logvar

    def sample(self, mu_tensor: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        This function samples from the Gaussian distribution
        :param mu: Mean of the Gaussian distribution
        :param logvar: Log variance of the Gaussian distribution
        :return: Sample from the Gaussian distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(
            std
        )  # check if this should add requires_grad = True to this tensor?
        return mu_tensor + eps * std

    def forward(self, text_embedding: torch.Tensor) -> Any:
        """
        This function encodes the text embedding into the conditioning space,
        and samples from the Gaussian distribution.
        :param text_embedding: Text embedding
        :return c_hat: Conditioning embedding (C^ from StackGAN++ paper)
        :return mu: Mean of the Gaussian distribution
        :return logvar: Log variance of the Gaussian distribution
        """
        mu_tensor, logvar = self.encode(text_embedding)
        c_hat = self.sample(mu_tensor, logvar)
        return c_hat, mu_tensor, logvar
