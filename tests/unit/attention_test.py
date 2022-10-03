from src.models.modules import ChannelWiseAttention

import torch

import pytest

@pytest.mark.parametrize(
    argnames=("batch_num", "C", "H_k", "W_k", "D", "L"),
    argvalues=(
            (32, 3,  16, 16, 5, 5),
            (16, 10, 3,  17, 3, 1),
            (2,  15, 5,  5,  1, 16),
    )
)
def test_channel_attention(batch_num, C, H_k, W_k, D, L):
    att = ChannelWiseAttention(H_k*W_k, D)
    # inputs
    v_k = torch.randn(batch_num, C, H_k * W_k)
    w = torch.randn(batch_num, D, L)
    f_k = att(v_k, w)
    assert f_k.size(1) == C and f_k.size(2) == H_k * W_k, "Input of visual features should not change"
