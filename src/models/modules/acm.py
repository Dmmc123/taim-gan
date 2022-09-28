"""ACM and its variations"""

from typing import Any

import torch
from torch import nn

from .conv_utils import conv


class ACM(nn.Module):
    """Affine Combination Module from ManiGAN"""

    def __init__(self, text_chans: int, img_chans: int, inner_dim: int = 64) -> None:
        """
        Initialize the convolutional layers

        :param int text_chans: Channels of textual input
        :param int img_chans: Channels in visual input
        :param int inner_dim: Hyperparameters for inner dimensionality of features
        """
        super().__init__()
        self.conv = conv(in_channels=img_chans, out_channels=inner_dim)
        self.weights = conv(in_channels=inner_dim, out_channels=text_chans)
        self.biases = conv(in_channels=inner_dim, out_channels=text_chans)

    def forward(self, text: torch.Tensor, img: torch.Tensor) -> Any:
        """
        Propagate the textual and visual input through the ACM module

        :param torch.Tensor text: Textual input
        :param torch.Tensor img: Image input
        :return: Affine combination of text and image
        :rtype: torch.Tensor
        """
        img_features = self.conv(img)
        return text * self.weights(img_features) + self.biases(img_features)
