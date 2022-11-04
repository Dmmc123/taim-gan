"""Generator Module"""

from typing import Any, Optional

import torch
from torch import nn

from src.models.modules.acm import ACM
from src.models.modules.attention import ChannelWiseAttention, SpatialAttention
from src.models.modules.cond_augment import CondAugmentation
from src.models.modules.downsample import down_sample
from src.models.modules.residual import ResidualBlock
from src.models.modules.upsample import img_up_block, up_sample


class InitStageG(nn.Module):
    """Initial Stage Generator Module"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals

    def __init__(
        self, Ng: int, Ng_init: int, conditioning_dim: int, D: int, noise_dim: int
    ):
        """
        :param Ng: Number of channels.
        :param Ng_init: Initial value of Ng, this is output channel of first image upsample.
        :param conditioning_dim: Dimension of the conditioning space
        :param D: Dimension of the text embedding space [D from AttnGAN paper]
        :param noise_dim: Dimension of the noise space
        """
        super().__init__()
        self.gf_dim = Ng
        self.gf_init = Ng_init
        self.in_dim = noise_dim + conditioning_dim + D
        self.text_dim = D

        self.define_module()

    def define_module(self) -> None:
        """Defines FC, Upsample, Residual, ACM, Attention modules"""
        nz, ng = self.in_dim, self.gf_dim
        self.fully_connect = nn.Sequential(
            nn.Linear(nz, ng * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ng * 4 * 4 * 2),
            nn.GLU(dim=1),  # we start from 4 x 4 feat_map and return hidden_64.
        )

        self.upsample1 = up_sample(ng, ng // 2)
        self.upsample2 = up_sample(ng // 2, ng // 4)
        self.upsample3 = up_sample(ng // 4, ng // 8)
        self.upsample4 = up_sample(
            ng // 8 * 3, ng // 16
        )  # multiply channel by 3 because concat spatial and channel att

        self.residual = self._make_layer(ResidualBlock, ng // 8 * 3)
        self.acm_module = ACM(self.gf_init, ng // 8 * 3)

        self.spatial_att = SpatialAttention(self.text_dim, ng // 8)
        self.channel_att = ChannelWiseAttention(
            32 * 32, self.text_dim
        )  # 32 x 32 is the feature map size

    def _make_layer(self, block: Any, channel_num: int) -> nn.Module:
        layers = []
        for _ in range(2):  # number of residual blocks hardcoded to 2
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        global_inception: torch.Tensor,
        local_upsampled_inception: torch.Tensor,
        word_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        :param noise: Noise tensor
        :param condition: Condition tensor (c^ from stackGAN++ paper)
        :param global_inception: Global inception feature
        :param local_upsampled_inception: Local inception feature, upsampled to 32 x 32
        :param word_embeddings: Word embeddings [shape: D x L or D x T]
        :param mask: Mask for padding tokens
        :return: Hidden Image feature map Tensor of 64 x 64 size
        """
        noise_concat = torch.cat((noise, condition), 1)
        inception_concat = torch.cat((noise_concat, global_inception), 1)
        hidden = self.fully_connect(inception_concat)
        hidden = hidden.view(-1, self.gf_dim, 4, 4)  # convert to 4x4 image feature map
        hidden = self.upsample1(hidden)
        hidden = self.upsample2(hidden)
        hidden_32 = self.upsample3(hidden)  # shape: (batch_size, gf_dim // 8, 32, 32)
        hidden_32_view = hidden_32.view(
            hidden_32.shape[0], -1, hidden_32.shape[2] * hidden_32.shape[3]
        )  # this reshaping is done as attention module expects this shape.

        spatial_att_feat = self.spatial_att(
            word_embeddings, hidden_32_view, mask
        )  # spatial att shape: (batch, D^, 32 * 32)
        channel_att_feat = self.channel_att(
            spatial_att_feat, word_embeddings
        )  # channel att shape: (batch, D^, 32 * 32), or (batch, C, Hk* Wk) from controlGAN paper
        spatial_att_feat = spatial_att_feat.view(
            word_embeddings.shape[0], -1, hidden_32.shape[2], hidden_32.shape[3]
        )  # reshape to (batch, D^, 32, 32)
        channel_att_feat = channel_att_feat.view(
            word_embeddings.shape[0], -1, hidden_32.shape[2], hidden_32.shape[3]
        )  # reshape to (batch, D^, 32, 32)

        spatial_concat = torch.cat(
            (hidden_32, spatial_att_feat), 1
        )  # concat spatial attention feature with hidden_32
        attn_concat = torch.cat(
            (spatial_concat, channel_att_feat), 1
        )  # concat channel and spatial attention feature

        hidden_32 = self.acm_module(attn_concat, local_upsampled_inception)
        hidden_32 = self.residual(hidden_32)
        hidden_64 = self.upsample4(hidden_32)
        return hidden_64


class NextStageG(nn.Module):
    """Next Stage Generator Module"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals

    def __init__(self, Ng: int, Ng_init: int, D: int, image_size: int):
        """
        :param Ng: Number of channels.
        :param Ng_init: Initial value of Ng.
        :param D: Dimension of the text embedding space [D from AttnGAN paper]
        :param image_size: Size of the output image from previous generator stage.
        """
        super().__init__()
        self.gf_dim = Ng
        self.gf_init = Ng_init
        self.text_dim = D
        self.img_size = image_size

        self.define_module()

    def define_module(self) -> None:
        """Defines FC, Upsample, Residual, ACM, Attention modules"""
        ng = self.gf_dim
        self.spatial_att = SpatialAttention(self.text_dim, ng)
        self.channel_att = ChannelWiseAttention(
            self.img_size * self.img_size, self.text_dim
        )

        self.residual = self._make_layer(ResidualBlock, ng * 3)
        self.upsample = up_sample(ng * 3, ng)
        self.acm_module = ACM(self.gf_init, ng * 3)
        self.upsample2 = up_sample(ng, ng)

    def _make_layer(self, block: Any, channel_num: int) -> nn.Module:
        layers = []
        for _ in range(2):  # no of residual layers hardcoded to 2
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(
        self,
        hidden_feat: Any,
        word_embeddings: torch.Tensor,
        vgg64_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        :param hidden_feat: Hidden feature from previous generator stage [i.e. hidden_64]
        :param word_embeddings: Word embeddings
        :param vgg64_feat: VGG feature map of size 64 x 64
        :param mask: Mask for the padding tokens
        :return: Image feature map of size 256 x 256
        """
        hidden_view = hidden_feat.view(
            hidden_feat.shape[0], -1, hidden_feat.shape[2] * hidden_feat.shape[3]
        )  # reshape to pass into attention modules.
        spatial_att_feat = self.spatial_att(
            word_embeddings, hidden_view, mask
        )  # spatial att shape: (batch, D^, 64 * 64), or D^ x N
        channel_att_feat = self.channel_att(
            spatial_att_feat, word_embeddings
        )  # channel att shape: (batch, D^, 64 * 64), or (batch, C, Hk* Wk) from controlGAN paper
        spatial_att_feat = spatial_att_feat.view(
            word_embeddings.shape[0], -1, hidden_feat.shape[2], hidden_feat.shape[3]
        )  # reshape to (batch, D^, 64, 64)
        channel_att_feat = channel_att_feat.view(
            word_embeddings.shape[0], -1, hidden_feat.shape[2], hidden_feat.shape[3]
        )  # reshape to (batch, D^, 64, 64)

        spatial_concat = torch.cat(
            (hidden_feat, spatial_att_feat), 1
        )  # concat spatial attention feature with hidden_64
        attn_concat = torch.cat(
            (spatial_concat, channel_att_feat), 1
        )  # concat channel and spatial attention feature

        hidden_64 = self.acm_module(attn_concat, vgg64_feat)
        hidden_64 = self.residual(hidden_64)
        hidden_128 = self.upsample(hidden_64)
        hidden_256 = self.upsample2(hidden_128)
        return hidden_256


class GetImageG(nn.Module):
    """Generates the Final Fake Image from the Image Feature Map"""

    def __init__(self, Ng: int):
        """
        :param Ng: Number of channels.
        """
        super().__init__()
        self.img = nn.Sequential(
            nn.Conv2d(Ng, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh()
        )

    def forward(self, hidden_feat: torch.Tensor) -> Any:
        """
        :param hidden_feat: Image feature map
        :return: Final fake image
        """
        return self.img(hidden_feat)


class Generator(nn.Module):
    """Generator Module"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals

    def __init__(self, Ng: int, D: int, conditioning_dim: int, noise_dim: int):
        """
        :param Ng: Number of channels. [Taken from StackGAN++ paper]
        :param D: Dimension of the text embedding space
        :param conditioning_dim: Dimension of the conditioning space
        :param noise_dim: Dimension of the noise space
        """
        super().__init__()
        self.cond_augment = CondAugmentation(D, conditioning_dim)
        self.hidden_net1 = InitStageG(Ng * 16, Ng, conditioning_dim, D, noise_dim)
        self.inception_img_upsample = img_up_block(
            D, Ng
        )  # as channel size returned by inception encoder is D (Default in paper: 256)
        self.hidden_net2 = NextStageG(Ng, Ng, D, 64)
        self.generate_img = GetImageG(Ng)

        self.acm_module = ACM(Ng, Ng)

        self.vgg_downsample = down_sample(D // 2, Ng)
        self.upsample1 = up_sample(Ng, Ng)
        self.upsample2 = up_sample(Ng, Ng)

    def forward(
        self,
        noise: torch.Tensor,
        sentence_embeddings: torch.Tensor,
        word_embeddings: torch.Tensor,
        global_inception_feat: torch.Tensor,
        local_inception_feat: torch.Tensor,
        vgg_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        :param noise: Noise vector [shape: (batch, noise_dim)]
        :param sentence_embeddings: Sentence embeddings [shape: (batch, D)]
        :param word_embeddings: Word embeddings [shape: D x L, where L is length of sentence]
        :param global_inception_feat: Global Inception feature map [shape: (batch, D)]
        :param local_inception_feat: Local Inception feature map [shape: (batch, D, 17, 17)]
        :param vgg_feat: VGG feature map [shape: (batch, D // 2 = 128, 128, 128)]
        :param mask: Mask for the padding tokens
        :return: Final fake image
        """
        c_hat, mu_tensor, logvar = self.cond_augment(sentence_embeddings)
        hidden_32 = self.inception_img_upsample(local_inception_feat)

        hidden_64 = self.hidden_net1(
            noise, c_hat, global_inception_feat, hidden_32, word_embeddings, mask
        )

        vgg_64 = self.vgg_downsample(vgg_feat)

        hidden_256 = self.hidden_net2(hidden_64, word_embeddings, vgg_64, mask)

        vgg_128 = self.upsample1(vgg_64)
        vgg_256 = self.upsample2(vgg_128)

        hidden_256 = self.acm_module(hidden_256, vgg_256)
        fake_img = self.generate_img(hidden_256)

        return fake_img, mu_tensor, logvar
