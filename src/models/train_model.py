"""Module to train the GAN model"""

from typing import Any

import torch

from src.models.losses import discriminator_loss, generator_loss, kl_loss
from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import Generator
from src.models.modules.image_encoder import InceptionEncoder, VGGEncoder
from src.models.modules.text_encoder import TextEncoder
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


def train(data_loader: Any, config_dict: dict[str, Any]) -> None:
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

    (
        optimizer_g,
        optimizer_d,
        optimizer_text_encoder,
        opt_image_encoder,
    ) = define_optimizers(
        generator, discriminator, image_encoder, text_encoder, lr_config
    )

    for epoch in range(1, epochs + 1):
        for batch_idx, (
            images,
            correct_capt,
            correct_capt_len,
            curr_class,
            word_labels,
        ) in enumerate(data_loader):

            labels_real, labels_fake, labels_match, fake_word_labels = prepare_labels(
                batch_size, word_labels.size(1), device
            )

            optimizer_d.zero_grad()

            noise = torch.randn(batch_size, noise_dim).to(device)
            word_emb, sent_emb = text_encoder(correct_capt)

            local_incept_feat, global_incept_feat = image_encoder(images)

            vgg_feat = vgg_encoder(images)
            mask = correct_capt == 0

            # Generate Fake Images
            fake_imgs, mu_tensor, logvar = generator(
                noise,
                sent_emb,
                word_emb,
                global_incept_feat,
                local_incept_feat,
                vgg_feat,
                mask,
            )

            # Generate Logits for discriminator update
            real_discri_feat = discriminator(images)
            fake_discri_feat = discriminator(fake_imgs.detach())

            logits_discri = {
                "fake": {
                    "uncond": discriminator.logits_uncond(fake_discri_feat),
                    "cond": discriminator.logits_cond(fake_discri_feat, sent_emb),
                },
                "real": {
                    "word_level": discriminator.logits_word_level(
                        real_discri_feat, word_emb, mask
                    ),
                    "uncond": discriminator.logits_uncond(real_discri_feat),
                    "cond": discriminator.logits_cond(real_discri_feat, sent_emb),
                },
            }

            labels_discri = {
                "fake": {"word_level": fake_word_labels, "image": labels_fake},
                "real": {"word_level": word_labels, "image": labels_real},
            }

            # Update Discriminator

            loss_discri = discriminator_loss(logits_discri, labels_discri)

            loss_discri.backward(retain_graph=True)
            optimizer_d.step()

            disc_loss.append(loss_discri.item())

            optimizer_g.zero_grad()

            word_emb, sent_emb = text_encoder(correct_capt)

            fake_imgs, mu_tensor, logvar = generator(
                noise,
                sent_emb,
                word_emb,
                global_incept_feat,
                local_incept_feat,
                vgg_feat,
                mask,
            )

            local_fake_incept_feat, global_fake_incept_feat = image_encoder(fake_imgs)

            vgg_feat_fake = vgg_encoder(fake_imgs)

            fake_feat_d = discriminator(fake_imgs)

            logits_gen = {
                "fake": {
                    "uncond": discriminator.logits_uncond(fake_feat_d),
                    "cond": discriminator.logits_cond(fake_feat_d, sent_emb),
                }
            }

            # Update Generator
            loss_gen = generator_loss(
                logits_gen,
                local_fake_incept_feat,
                global_fake_incept_feat,
                labels_real,
                word_emb,
                sent_emb,
                labels_match,
                correct_capt_len,
                curr_class,
                vgg_feat,
                vgg_feat_fake,
                const_dict,
            )

            loss_kl = kl_loss(mu_tensor, logvar)

            loss_gen += loss_kl

            loss_gen.backward()
            optimizer_g.step()
            gen_loss.append(loss_gen.item())

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}],\
                    Loss D: {loss_discri.item():.4f}, Loss G: {loss_gen.item():.4f}"
                )

            if (batch_idx + 1) % 50 == 0:
                with torch.no_grad():
                    fake_imgs_act, _, _ = generator(
                        noise,
                        sent_emb,
                        word_emb,
                        global_incept_feat,
                        local_incept_feat,
                        vgg_feat,
                        mask,
                    )
                    save_image_and_caption(
                        fake_imgs_act,
                        images,
                        correct_capt,
                        ix2word,
                        batch_idx,
                        epoch,
                        output_dir,
                    )
                    save_plot(gen_loss, disc_loss, epoch, batch_idx, output_dir)

        if epoch % snapshot == 0 and epoch != 0:
            save_model(
                generator, discriminator, image_encoder, text_encoder, epoch, output_dir
            )

    save_model(
        generator, discriminator, image_encoder, text_encoder, epochs, output_dir
    )
