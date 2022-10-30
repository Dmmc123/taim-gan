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

    def forward(
        self, visual_features: torch.Tensor, word_embs: torch.Tensor
    ) -> Any:
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

    def __init__(self, n_words: int) -> None:
        """
        Initialize modules that reduce the features down to a set of logits

        :param int n_words: Max length of a caption
        """
        super().__init__()
        self.squisher = nn.Sequential(
            conv2d(128, n_words),  # for reducing the channel dimension
            nn.Conv2d(n_words, n_words, kernel_size=17),  # to reduce feature maps
        )
        # flattening BxLx1x1 into BxL
        self.flat = nn.Flatten()

    def forward(self, visual_features: torch.Tensor) -> Any:
        """
        Compute logits for unconditioned adversarial loss

        :param visual_features: Local features from Inception network. Bx128x17x17
        :return: Logits for unconditioned adversarial loss. BxL
        :rtype: Any
        """
        # reduce channels and feature maps for visual features
        visual_features = self.squisher(visual_features)
        # flatten BxLx1x1 into BxL
        logits = self.flat(visual_features)
        return logits


class ConditionalLogits(nn.Module):
    """Logits extractor for conditioned adversarial loss"""

    def __init__(self, n_words: int) -> None:
        super().__init__()
        # recording value of L
        self.n_words = n_words
        # layer for increasing the number of textual channels
        self.text_conv = conv1d(256, 17 * 17)
        # for reduced textual + visual features down to 1x1 feature map
        self.joint_conv = nn.Conv2d(128 + n_words, n_words, kernel_size=17)
        # converting BxLx1x1 into BxL
        self.flat = nn.Flatten()

    def forward(self, visual_features: torch.Tensor, sent_embs: torch.Tensor) -> Any:
        """
        Compute logits for conditional adversarial loss

        :param torch.Tensor visual_features: Features from Inception encoder. Bx128x17x17
        :param torch.Tensor sent_embs: Sentence embeddings from text encoder. Bx256
        :return: Logits for conditional adversarial loss. BxL
        :rtype: Any
        """
        # propagate text embs through 1d conv to
        # align dims with visual feature maps
        sent_embs = self.text_conv(sent_embs)
        # transform textual info into shape of visual feature maps
        # Bx289xL -> BxLx289 -> BxLx17x17
        sent_embs = torch.transpose(sent_embs, 1, 2)
        sent_embs = sent_embs.view(-1, self.n_words, 17, 17)
        # unite textual and visual features across the dim of channels
        cross_features = torch.cat((visual_features, sent_embs), dim=1)
        # reduce dims down to length of caption and form raw logits
        cross_features = self.joint_conv(cross_features)
        # form logits from BxLx1x1 into BxL
        logits = self.flat(cross_features)
        return logits


class Discriminator(nn.Module):
    """Simple CNN-based discriminator"""

    def __init__(self, n_words: int) -> None:
        """
        Use a pretrained InceptionNet to extract features

        :param int n_words: Max length of a text caption for an image
        """
        super().__init__()
        self.encoder = InceptionEncoder(D=128)
        # define different logit extractors for different losses
        self.logits_word_level = WordLevelLogits()
        self.logits_uncond = UnconditionalLogits(n_words=n_words)
        self.logits_cond = ConditionalLogits(n_words=n_words)

    def forward(self,
                images: torch.Tensor,
                sent_embs: torch.Tensor,
                word_embs: torch.Tensor) -> Any:
        """
        Obtain regional features for images and return logits

        :param torch.Tensor images: Images to be analyzed. Bx3x256x256
        :param torch.Tensor sent_embs: Sentence-level embeddings from text encoder Bx256
        :param torch.Tensor word_embs: Word-level embeddings from text encoder Bx256xL
        :return: Types of logits for different losses. BxL
        :rtype: Any
        """
        # only taking the local features from inception
        # Bx3x256x256 -> Bx128x17x17
        img_features, _ = self.encoder(images)
        # getting word-level feedback for the generated image
        logits_word_level = self.logits_word_level(img_features, word_embs)
        # getting unconditioned adversarial logits
        logits_uncond = self.logits_uncond(img_features)
        # computing logits for loss conditioned on sentence-level embeddings
        logits_cond = self.logits_cond(img_features, sent_embs)
        return logits_word_level, logits_uncond, logits_cond
