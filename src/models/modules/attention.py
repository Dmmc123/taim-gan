"""Attention modules"""
from typing import Any

import torch
from torch import nn

from src.models.modules.conv_utils import conv1d


class ChannelWiseAttention(nn.Module):
    """ChannelWise attention adapted from ControlGAN"""

    def __init__(self, fm_size: int, text_d: int) -> None:
        """
        Initialize the Channel-Wise attention module

        :param int fm_size:
            Height and width of feature map on k-th iteration of forward-pass.
            In paper, it's H_k * W_k
        :param int text_d: Dimensionality of sentence. From paper, it's D
        """
        super().__init__()
        # perception layer
        self.text_conv = conv1d(text_d, fm_size)
        # attention across channel dimension
        self.softmax = nn.Softmax(1)

    def forward(self, v_k: torch.Tensor, w_text: torch.Tensor) -> Any:
        """
        Apply attention to visual features taking into account features of words

        :param torch.Tensor v_k: Visual context
        :param torch.Tensor w_text: Textual features
        :return: Fused hidden visual features and word features
        :rtype: Any
        """
        w_hat = self.text_conv(w_text)
        m_k = v_k @ w_hat
        a_k = self.softmax(m_k)
        w_hat = torch.transpose(w_hat, 1, 2)
        return a_k @ w_hat


class SpatialAttention(nn.Module):
    """Spatial attention module for attending textual context to visual features"""

    def __init__(self, d: int, d_hat: int) -> None:
        """
        Set up softmax and conv layers

        :param int d: Initial embedding size for textual features. D from paper
        :param int d_hat: Height of image feature map. D_hat from paper
        """
        super().__init__()
        self.softmax = nn.Softmax(1)
        self.conv = conv1d(d, d_hat)

    def forward(self, text_context: torch.Tensor, image: torch.Tensor) -> Any:
        """
        Project image features into the latent space
        of textual features and apply attention

        :param text_context: D x T tensor of hidden textual features
        :param image: D_hat x N visual features
        :return: Word features attended by visual features
        :rtype: Any
        """
        text_context = self.conv(text_context)
        image = torch.transpose(image, 1, 2)
        s_i_j = image @ text_context
        b_i_j = self.softmax(s_i_j)
        c_i_j = b_i_j @ torch.transpose(text_context, 1, 2)
        return torch.transpose(c_i_j, 1, 2)
