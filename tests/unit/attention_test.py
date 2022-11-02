from src.models.modules import ChannelWiseAttention, SpatialAttention

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


@pytest.mark.parametrize(
    argnames=("batch_num", "D", "T", "D_hat", "N"),
    argvalues=(
            (64, 3, 5, 16, 16),
            (8,  7, 3, 28, 7),
            (16, 5, 5, 5,  1),
    )
)
def test_spatial_attention(batch_num, D, T, D_hat, N):
    att = SpatialAttention(D, D_hat)
    # inputs
    e = torch.randn(batch_num, D, T)
    h = torch.randn(batch_num, D_hat, N)
    mask = torch.randint(0, 2, (batch_num, T)).bool()
    c = att(e, h, mask=mask)
    assert c.size(1) == D_hat and c.size(2) == N, "Input of visual features should not change"
