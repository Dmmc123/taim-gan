"""Residual Block Adopted from ManiGAN"""

from typing import Any

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, channel_num: int) -> None:
        """
        :param channel_num: Number of channels in the input
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channel_num,
                channel_num * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(channel_num * 2),
            nn.GLU(dim=1),
            nn.Conv2d(
                channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(channel_num),
        )

    def forward(self, input_tensor: torch.Tensor) -> Any:
        """
        :param input_tensor: Input tensor
        :return: Output tensor
        """
        residual = input_tensor
        out = self.block(input_tensor)
        out += residual
        return out
