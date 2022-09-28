"""Frequently used convolution modules"""

from torch import nn


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
) -> nn.Conv2d:
    """
    Template convolution which is typically used throughout the project

    :param int in_channels: Number of input channels
    :param int out_channels: Number of output channels
    :param int kernel_size: Size of sliding kernel
    :param int stride: How many steps kernel does when sliding
    :param int padding: How many dimensions to pad
    :return: Convolution layer with parameters
    :rtype: nn.Conv2d
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def calc_out_conv(
    h_in: int, w_in: int, kernel_size: int = 3, stride: int = 1, padding: int = 0
) -> tuple[int, int]:
    """
    Calculate the dimensionalities of images propagated through conv layers

    :param h_in: Height of the image
    :param w_in: Width of the image
    :param kernel_size: Size of sliding kernel
    :param stride: How many steps kernel does when sliding
    :param padding: How many dimensions to pad
    :return: Height and width of image through convolution
    :rtype: tuple[int, int]
    """
    h_out = int((h_in + 2 * padding - kernel_size) / stride + 1)
    w_out = int((w_in + 2 * padding - kernel_size) / stride + 1)
    return h_out, w_out
