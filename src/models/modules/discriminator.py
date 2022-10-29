"""Discriminator providing word-level feedback"""
from typing import Any

import torch
from torch import nn

from src.models.modules.conv_utils import conv1d
from src.models.modules.image_encoder import InceptionEncoder


class WordLevelLogits(nn.Module):
    """API for converting regional feature maps into logits for multi-class classification"""

    def __init__(self) -> None:
        """
        Instantiate the module with softmax on channel dimension
        """
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        # layer for flattening the feature maps
        self.flat = nn.Flatten(start_dim=2)
        # change dism of of textual embs to correlate with chans of inception
        self.chan_reduction = conv1d(256, 128)

    def forward(
        self, visual_features: torch.Tensor, textual_features: torch.Tensor
    ) -> Any:
        """
        Fuse two types of features together to get output for feeding into the classification loss
        :param torch.Tensor visual_features:
            Feature maps of an image after being processed by discriminator. Bx128x17x17
        :param torch.Tensor textual_features: Result of text encoder Bx256xL
        :return: Logits for each word in the picture. BxL
        :rtype: Any
        """
        # make textual and visual features have the same amount of channels
        textual_features = self.chan_reduction(textual_features)
        # flattening the feature maps
        visual_features = self.flat(visual_features)
        textual_features = torch.transpose(textual_features, 1, 2)
        word_region_correlations = textual_features @ visual_features
        # normalize across L dimension
        m_norm_l = nn.functional.normalize(word_region_correlations, dim=1)
        # normalize across H*W dimension
        m_norm_hw = nn.functional.normalize(m_norm_l, dim=2)
        m_norm_hw = torch.transpose(m_norm_hw, 1, 2)
        weighted_img_feats = visual_features @ m_norm_hw
        weighted_img_feats = torch.sum(weighted_img_feats, dim=1)
        deltas = self.softmax(weighted_img_feats)
        return deltas


# class UnconditionalLogits(nn.Module):
#     """Head for retrieving logits from an image"""
#
#     def __init__(self):
#         super().__init__()


class Discriminator(nn.Module):
    """Simple CNN-based discriminator"""

    def __init__(self) -> None:
        """Use a pretrained InceptionNet to extract features"""
        super().__init__()
        self.encoder = InceptionEncoder(D=128)
        # skip batch and channel dims, flatten only feature maps
        self.flat_fm = nn.Flatten(start_dim=2)
        self.logits_word_level = WordLevelLogits()

    def forward(self, images: torch.Tensor, textual_info: torch.Tensor) -> Any:
        """
        Obtain regional features for images and return logits

        :param images: Images to be analyzed. Bx3x256x256
        :param textual_info: Output of RNN (text encoder). Bx3xL
        :return: Types of logits for different losses
        :rtype: Any
        """
        # only taking the local features from inception
        img_features, _ = self.encoder(images)
        # getting word-level feedback for the generated image
        logits_word_level = self.logits_word_level(img_features, textual_info)
        return logits_word_level


# def main():
#     img_size = (3, 256, 256)
#     batch = 4
#     hidden_dim = 256
#     n_words = 18
#
#     D = Discriminator()
#
#     images = torch.rand((batch, *img_size))
#     textual_info = torch.rand((batch, hidden_dim, n_words))
#
#     logits = D(images, textual_info)
#     print(logits.size())
#
#
# if __name__ == "__main__":
#     main()
