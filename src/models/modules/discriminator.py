"""Discriminator providing word-level feedback"""
from typing import Any

import torch
from torch import nn

from src.models.modules.conv_utils import conv1d, conv2d
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

    def forward(self, visual_features: torch.Tensor, word_embs: torch.Tensor) -> Any:
        """
        Fuse two types of features together to get output for feeding into the classification loss
        :param torch.Tensor visual_features:
            Feature maps of an image after being processed by Inception encoder. Bx128x17x17
        :param torch.Tensor word_embs:
            Word-level embeddings from the text encoder Bx256xL
        :return: Logits for each word in the picture. BxL
        :rtype: Any
        """
        # make textual and visual features have the same amount of channels
        word_embs = self.chan_reduction(word_embs)
        # flattening the feature maps
        visual_features = self.flat(visual_features)
        word_embs = torch.transpose(word_embs, 1, 2)
        word_region_correlations = word_embs @ visual_features
        # normalize across L dimension
        m_norm_l = nn.functional.normalize(word_region_correlations, dim=1)
        # normalize across H*W dimension
        m_norm_hw = nn.functional.normalize(m_norm_l, dim=2)
        m_norm_hw = torch.transpose(m_norm_hw, 1, 2)
        weighted_img_feats = visual_features @ m_norm_hw
        weighted_img_feats = torch.sum(weighted_img_feats, dim=1)
        deltas = self.softmax(weighted_img_feats)
        return deltas


class UnconditionalLogits(nn.Module):
    """Head for retrieving logits from an image"""

    def __init__(self) -> None:
        """Initialize modules that reduce the features down to a set of logits"""
        super().__init__()
        self.conv = nn.Conv2d(128, 1, kernel_size=17)
        # flattening BxLx1x1 into Bx1
        self.flat = nn.Flatten()

    def forward(self, visual_features: torch.Tensor) -> Any:
        """
        Compute logits for unconditioned adversarial loss

        :param visual_features: Local features from Inception network. Bx128x17x17
        :return: Logits for unconditioned adversarial loss. Bx1
        :rtype: Any
        """
        # reduce channels and feature maps for visual features
        visual_features = self.conv(visual_features)
        # flatten Bx1x1x1 into Bx1
        logits = self.flat(visual_features)
        return logits


class ConditionalLogits(nn.Module):
    """Logits extractor for conditioned adversarial loss"""

    def __init__(self) -> None:
        super().__init__()
        # layer for forming the feature maps out of textual info
        self.text_to_fm = conv1d(256, 17 * 17)
        # fitting the size of text channels to the size of visual channels
        self.chan_aligner = conv2d(1, 128)
        # for reduced textual + visual features down to 1x1 feature map
        self.joint_conv = nn.Conv2d(2 * 128, 1, kernel_size=17)
        # converting Bx1x1x1 into Bx1
        self.flat = nn.Flatten()

    def forward(self, visual_features: torch.Tensor, sent_embs: torch.Tensor) -> Any:
        """
        Compute logits for conditional adversarial loss

        :param torch.Tensor visual_features: Features from Inception encoder. Bx128x17x17
        :param torch.Tensor sent_embs: Sentence embeddings from text encoder. Bx256
        :return: Logits for conditional adversarial loss. BxL
        :rtype: Any
        """
        # make text and visual features have the same sizes of feature maps
        # Bx256 -> Bx256x1 -> Bx289x1
        sent_embs = sent_embs.view(-1, 256, 1)
        sent_embs = self.text_to_fm(sent_embs)
        # transform textual info into shape of visual feature maps
        # Bx289x1 -> Bx1x17x17
        sent_embs = sent_embs.view(-1, 1, 17, 17)
        # propagate text embs through 1d conv to
        # align dims with visual feature maps
        sent_embs = self.chan_aligner(sent_embs)
        # unite textual and visual features across the dim of channels
        cross_features = torch.cat((visual_features, sent_embs), dim=1)
        # reduce dims down to length of caption and form raw logits
        cross_features = self.joint_conv(cross_features)
        # form logits from Bx1x1x1 into Bx1
        logits = self.flat(cross_features)
        return logits


class Discriminator(nn.Module):
    """Simple CNN-based discriminator"""

    def __init__(self) -> None:
        """Use a pretrained InceptionNet to extract features"""
        super().__init__()
        self.encoder = InceptionEncoder(D=128)
        # define different logit extractors for different losses
        self.logits_word_level = WordLevelLogits()
        self.logits_uncond = UnconditionalLogits()
        self.logits_cond = ConditionalLogits()

    def forward(self, images: torch.Tensor) -> Any:
        """
        Retrieves image features encoded by the image encoder

        :param torch.Tensor images: Images to be analyzed. Bx3x256x256
        :return: image features encoded by image encoder. Bx128x17x17
        """
        # only taking the local features from inception
        # Bx3x256x256 -> Bx128x17x17
        img_features, _ = self.encoder(images)
        return img_features
