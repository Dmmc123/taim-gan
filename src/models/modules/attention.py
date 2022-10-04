"""Attention modules"""
from typing import Any

import torch
from torch import nn

from src.models.modules.conv_utils import conv2d


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
        self.text_conv = conv2d(text_d, fm_size, kernel_size=1)
        # attention across channel dimension
        self.softmax = nn.Softmax(1)
        # dimensionalities of output feature maps
        self.fm_size = fm_size
        self.text_d = text_d

    def forward(self, v_k: torch.Tensor, w_text: torch.Tensor) -> Any:
        """
        Apply attention to visual features taking into account features of words

        :param torch.Tensor v_k: Visual context
        :param torch.Tensor w_text: Textual features
        :return: Fused hidden visual features and word features
        :rtype: Any
        """
        # get sqrt of L from input
        emb_l = w_text.size(2)
        # convert embs to 2d feature maps
        w_text = w_text.view(-1, self.text_d, emb_l, 1)
        w_hat = self.text_conv(w_text)
        # return to flat embs
        w_hat = w_hat.view(-1, self.fm_size, emb_l)
        m_k = v_k @ w_hat
        a_k = self.softmax(m_k)
        w_hat = torch.transpose(w_hat, 1, 2)
        return a_k @ w_hat
