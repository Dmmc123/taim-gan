"""Image Encoder Module"""
import torch.nn.functional as F
from torch import nn
from torch.utils import model_zoo
from torchvision import models

from src.models.modules.conv_utils import conv2d

# build inception v3 image encoder


class ImageEncoder(nn.Module):
    """Image Encoder Module adapted from AttnGAN"""

    def __init__(self, D: int):
        """
        :param D: Dimension of the text embedding space [D from AttnGAN paper]
        """
        super().__init__()

        self.nef = D

        model = models.inception_v3(init_weights=True)
        url = "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print("Load pretrained model from ", url)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        """
        This function defines the modules of the image encoder
        :param model: Pretrained Inception V3 model
        """
        self.conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.mixed_5b = model.Mixed_5b
        self.mixed_5c = model.Mixed_5c
        self.mixed_5d = model.Mixed_5d
        self.mixed_6a = model.Mixed_6a
        self.mixed_6b = model.Mixed_6b
        self.mixed_6c = model.Mixed_6c
        self.mixed_6d = model.Mixed_6d
        self.mixed_6e = model.Mixed_6e
        self.mixed_7a = model.Mixed_7a
        self.mixed_7b = model.Mixed_7b
        self.mixed_7c = model.Mixed_7c

        self.emb_features = conv2d(768, self.nef, kernel_size=1)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        """
        This function initializes the trainable weights of the image encoder
        """
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """
        :param x: Input image
        :return: features: local feature matrix (v from attnGAN paper)
        :return: cnn_code: global image feature (v^ from attnGAN paper)
        """

        # this is the image size
        # x.shape: 10 3 256 256

        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear")(x)
        # 299 x 299 x 3
        x = self.conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.mixed_5b(x)
        # 35 x 35 x 256
        x = self.mixed_5c(x)
        # 35 x 35 x 288
        x = self.mixed_5d(x)
        # 35 x 35 x 288

        x = self.mixed_6a(x)
        # 17 x 17 x 768
        x = self.mixed_6b(x)
        # 17 x 17 x 768
        x = self.mixed_6c(x)
        # 17 x 17 x 768
        x = self.mixed_6d(x)
        # 17 x 17 x 768
        x = self.mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.mixed_7a(x)
        # 8 x 8 x 1280
        x = self.mixed_7b(x)
        # 8 x 8 x 2048
        x = self.mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)

        # feature.shape: 10 256 17 17
        # cnn_code.shape: 10 256
        return features, cnn_code
