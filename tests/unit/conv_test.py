from src.models.modules import conv, calc_out_conv

import torch

import pytest


@pytest.mark.parametrize(
    argnames=("batch_num", "chans", "h_in", "w_in", "out_chans", "kernel", "stride", "padding"),
    argvalues=(
        (64,  3, 100, 100, 3,  3, 1, 1),
        (128, 5, 50,  256, 1,  5, 2, 0),
        (256, 7, 128, 64,  10, 9, 3, 5),
    )
)
def test_conv(batch_num, chans, h_in, w_in, out_chans, kernel, stride, padding):
    x = torch.rand(size=(batch_num, chans, h_in, w_in))
    conv_layer = conv(
        in_channels=chans,
        out_channels=out_chans,
        kernel_size=kernel,
        stride=stride,
        padding=padding
    )
    x = conv_layer(x)
    h, w = list(x.size())[-2:]
    exp_h, exp_w = calc_out_conv(
        h_in=h_in,
        w_in=w_in,
        kernel_size=kernel,
        stride=stride,
        padding=padding
    )
    assert exp_h == h and exp_w == w, "Calculation of dimensions should work fine"
