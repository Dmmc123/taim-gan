"""Image Encoder Module"""
from typing import Any

import torch
from torch import nn

from src.models.modules.conv_utils import conv2d

# build inception v3 image encoder


class InceptionEncoder(nn.Module):
    """Image Encoder Module adapted from AttnGAN"""

    def __init__(self, D: int):
        """
        :param D: Dimension of the text embedding space [D from AttnGAN paper]
        """
        super().__init__()

        self.text_emb_dim = D

        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
        )
        for param in model.parameters():
            param.requires_grad = False

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model: nn.Module) -> None:
        """
        This function defines the modules of the image encoder
        :param model: Pretrained Inception V3 model
        """
        model.cust_upsample = nn.Upsample(size=(299, 299), mode="bilinear")
        model.cust_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        model.cust_maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        model.cust_avgpool = nn.AvgPool2d(kernel_size=8)

        attribute_list = [
            "cust_upsample",
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            "cust_maxpool1",
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            "cust_maxpool2",
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
        ]

        self.feature_extractor = nn.Sequential(
            *[getattr(model, name) for name in attribute_list]
        )

        attribute_list2 = ["Mixed_7a", "Mixed_7b", "Mixed_7c", "cust_avgpool"]

        self.feature_extractor2 = nn.Sequential(
            *[getattr(model, name) for name in attribute_list2]
        )

        self.emb_features = conv2d(
            768, self.text_emb_dim, kernel_size=1, stride=1, padding=0
        )
        self.emb_cnn_code = nn.Linear(2048, self.text_emb_dim)

    def init_trainable_weights(self) -> None:
        """
        This function initializes the trainable weights of the image encoder
        """
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, image_tensor: torch.Tensor) -> Any:
        """
        :param image_tensor: Input image
        :return: features: local feature matrix (v from attnGAN paper) [shape: (batch, D, 17, 17)]
        :return: cnn_code: global image feature (v^ from attnGAN paper) [shape: (batch, D)]
        """
        # this is the image size
        # x.shape: 10 3 256 256

        features = self.feature_extractor(image_tensor)
        # 17 x 17 x 768

        image_tensor = self.feature_extractor2(features)

        image_tensor = image_tensor.view(image_tensor.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(image_tensor)

        if features is not None:
            features = self.emb_features(features)

        # feature.shape: 10 256 17 17
        # cnn_code.shape: 10 256
        return features, cnn_code


class VGGEncoder(nn.Module):
    """Pre Trained VGG Encoder Module"""

    def __init__(self) -> None:
        """
        Initialize pre-trained VGG model with frozen parameters
        """
        super().__init__()
        self.select = "8"  ## We want to get the output of the 8th layer in VGG.

        model = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True)

        for param in model.parameters():
            param.resquires_grad = False

        self.vgg_modules = model.features._modules

    def forward(self, image_tensor: torch.Tensor) -> Any:
        """
        :param x: Input image tensor [shape: (batch, 3, 256, 256)]
        :return: VGG features [shape: (batch, 128, 128, 128)]
        """
        for name, layer in self.vgg_modules.items():
            image_tensor = layer(image_tensor)
            if name in self.select:
                return image_tensor
        return None
