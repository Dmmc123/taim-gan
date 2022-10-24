"""Discriminator providing word-level feedback"""
from typing import Any

import torch
from torch import nn

from src.models.modules.image_encoder import InceptionEncoder


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


class Discriminator(nn.Module):
    """Simple CNN-based discriminator"""

    def __init__(self, emb_len: int) -> None:
        """
        Use a pretrained InceptionNet to extract features

        :param int emb_len: Length of textual embedding
        """
        super().__init__()
        self.encoder = InceptionEncoder(D=emb_len)
        # skip batch and channel dims, flatten only feature maps
        self.flat = nn.Flatten(start_dim=2)
        self.logits = FeedbackModule()

    def forward(self, images: torch.Tensor, textual_info: torch.Tensor) -> Any:
        """
        Obtain regional features for images and return word logits from that image
        :param images: Images to be analyzed
        :param textual_info: Output of RNN (text encoder)
        :return: Word-level feedback (logits) for presence of text in picture
        :rtype: Any
        """
        # only taking the local features from inception
        img_features, _ = self.encoder(images)
        # flattening the feature maps into BxCx(H*W)
        img_flat = self.flat(img_features)
        # getting word-level feedback for the generated image
        logits = self.logits(img_flat, textual_info)
        return logits
