"""Module to train the GAN model"""

from typing import Any

import torch
import numpy as np
from src.models.losses import discriminator_loss, generator_loss, kl_loss
from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import Generator
from src.models.modules.image_encoder import InceptionEncoder, VGGEncoder
from src.models.modules.text_encoder import TextEncoder
from pytorch_gan_metrics import get_inception_score
from src.models.utils import (
    define_optimizers,
    load_model,
    prepare_labels,
    save_image_and_caption,
    save_model,
    save_plot,
)

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements


def inception_score_from_probs(probs, eps=1e-16):
    # compute average probabilities of classes
    mean_probs = torch.mean(probs, dim=0, keepdim=True)
    # compute KL-divergence
    kl_d = probs * (torch.log(probs + eps) - torch.log(mean_probs + eps))
    # sum over classes, average over images, and undo logs
    score = torch.exp(torch.mean(torch.nansum(kl_d, dim=1)))
    return score


def compute_metrics(data_loader: Any, config_dict: dict[str, Any]) -> None:
    """
    Function to train the GAN model
    :param data_loader: Data loader for the dataset
    :param vocab_len: Length of the vocabulary
    :param config_dict: Dictionary containing the configuration parameters
    """
    (
        Ng,  # pylint: disable=invalid-name
        D,  # pylint: disable=invalid-name
        condition_dim,
        noise_dim,
        lr_config,
        batch_size,
        device,
        epochs,
        vocab_len,
        ix2word,
        output_dir,
        snapshot,
        const_dict,
    ) = (
        config_dict["Ng"],
        config_dict["D"],
        config_dict["condition_dim"],
        config_dict["noise_dim"],
        config_dict["lr_config"],
        config_dict["batch_size"],
        config_dict["device"],
        config_dict["epochs"],
        config_dict["vocab_len"],
        config_dict["ix2word"],
        config_dict["output_dir"],
        config_dict["snapshot"],
        config_dict["const_dict"],
    )

    generator = Generator(Ng, D, condition_dim, noise_dim).to(device)
    discriminator = Discriminator().to(device)
    text_encoder = TextEncoder(vocab_len, D, D // 2).to(device)
    image_encoder = InceptionEncoder(D).to(device)
    vgg_encoder = VGGEncoder().to(device)
    gen_loss = []
    disc_loss = []

    load_model(generator, discriminator, image_encoder, text_encoder, output_dir)

    for param in image_encoder.parameters():
        param.requires_grad = False

    for param in text_encoder.parameters():
        param.requires_grad = False

    for param in generator.parameters():
        param.requires_grad = False

    for param in discriminator.parameters():
        param.requires_grad = False

    epochs = 1

    # classifier network
    inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)

    # go through all testing data and get probabilities from inception
    probs = []

    for epoch in range(1, epochs + 1):
        for batch_idx, (
            images,
            correct_capt,
            correct_capt_len,
            curr_class,
            word_labels,
        ) in enumerate(data_loader):


            with torch.no_grad():
                noise = torch.randn(batch_size, noise_dim).to(device)
                word_emb, sent_emb = text_encoder(correct_capt)

                local_incept_feat, global_incept_feat = image_encoder(images)

                vgg_feat = vgg_encoder(images)
                mask = correct_capt == 0
                generated_images, _, _ = generator(
                    noise,
                    sent_emb,
                    word_emb,
                    global_incept_feat,
                    local_incept_feat,
                    vgg_feat,
                    mask,
                )
                # get probabilities from inception
                
                generated_images = torch.nn.functional.interpolate(
                    generated_images,
                    size=(299, 299),
                    mode="bilinear",
                    align_corners=False,
                )

                probs.append( torch.nn.Softmax(dim = 1)(inception(generated_images).logits) )

    all_probs = torch.cat(probs, dim=0)
    inception_score = inception_score_from_probs(all_probs)

    print(f"Inception score for UTKFace: {inception_score}")

                
