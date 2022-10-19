"""Discriminator providing word-level feedback"""
from typing import Any

import torch
from torch import nn


class FeedbackModule(nn.Module):
    """API for converting regional feature maps into logits for multi-class classification"""

    def __init__(self) -> None:
        """
        Instantiate the module with softmax on channel dimension
        """
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> Any:
        """
        Fuse two types of features together to get output for feeding into the classification loss
        :param torch.Tensor visual_features:
            Feature maps of an image after being processed by discriminator
        :param torch.Tensor textual_features: Result of text encoder
        :return: Logits for each word in the picture
        :rtype: Any
        """
        visual_features = torch.transpose(visual_features, 1, 2)
        word_region_correlations = visual_features @ textual_features
        # normalize across L dimension
        m_norm_l = nn.functional.normalize(word_region_correlations, dim=1)
        # normalize across H*W dimension
        m_norm_hw = nn.functional.normalize(m_norm_l, dim=2)
        m_norm_hw = torch.transpose(m_norm_hw, 1, 2)
        weighted_img_feats = textual_features @ m_norm_hw
        weighted_img_feats = torch.sum(weighted_img_feats, dim=1)
        deltas = self.softmax(weighted_img_feats)
        return deltas


class Discriminator(nn.Module):
    """Simple CNN-based discriminator"""

    def __init__(self) -> None:
        """
        Create a bunch of convolutions to extract features
        """
        super().__init__()
        self.convs = nn.Sequential(
            *[
                self.conv_block(in_chans, out_chans)
                for in_chans, out_chans in [
                    (3, 5),
                    (5, 7),
                    (7, 9),
                    (9, 7),
                    (7, 5),
                    (5, 3),
                ]
            ]
        )
        # skip batch and channel dims, flatten only feature maps
        self.flat = nn.Flatten(start_dim=2)
        self.logits = FeedbackModule()

    @staticmethod
    def conv_block(in_chans: int, out_chans: int) -> nn.Sequential:
        """
        Simple feature extraction block followed by an activation function
        :param int in_chans: Number of input channels for conv layer
        :param int out_chans: Number of output channels for conv layer
        :return: Reusable convolutional block to extract features
        :rtype: Any
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, images: torch.Tensor, textual_info: torch.Tensor) -> Any:
        """
        Obtain regional features for images and return word logits from that image
        :param images: Images to be analyzed
        :param textual_info: Output of RNN (text encoder)
        :return: Word-level feedback (logits) for presence of text in picture
        :rtype: Any
        """
        img_features = self.convs(images)
        img_flat = self.flat(img_features)
        logits = self.logits(img_flat, textual_info)
        return logits
