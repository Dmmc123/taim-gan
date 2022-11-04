"""Module to train the GAN model"""

from typing import Any

import torch

from src.models.losses import discriminator_loss, generator_loss, kl_loss
from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import Generator
from src.models.modules.image_encoder import InceptionEncoder, VGGEncoder
from src.models.modules.text_encoder import TextEncoder
from src.models.utils import (
    copy_gen_params,
    define_optimizers,
    load_params,
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

    smooth_val_gen = const_dict["smooth_val_gen"]
    lambda4 = const_dict["lambda4"]
    generator = Generator(Ng, D, condition_dim, noise_dim).to(device)
    discriminator = Discriminator().to(device)
    text_encoder = TextEncoder(vocab_len, D, D // 2).to(device)
    image_encoder = InceptionEncoder(D).to(device)
    vgg_encoder = VGGEncoder().to(device)
    gen_loss = []
    disc_loss = []

    g_param_avg = copy_gen_params(generator)

    optimizer_g, optimizer_d, optimizer_text_encoder = define_optimizers(
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

            local_fake_incept_feat, global_fake_incept_feat = image_encoder(fake_imgs)
            vgg_feat_fake = vgg_encoder(fake_imgs)

            # Generate Logits for discriminator update
            real_discri_feat = discriminator(images)
            fake_discri_feat = discriminator(fake_imgs)

            logits_discri = {
                "fake": {
                    "word_level": discriminator.logits_word_level(
                        fake_discri_feat, word_emb
                    ),
                    "uncond": discriminator.logits_uncond(fake_discri_feat),
                    "cond": discriminator.logits_cond(fake_discri_feat, sent_emb),
                },
                "real": {
                    "word_level": discriminator.logits_word_level(
                        real_discri_feat, word_emb
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
            optimizer_d.zero_grad()
            loss_discri = discriminator_loss(logits_discri, labels_discri, lambda4)

            loss_discri.backward(retain_graph=True)
            optimizer_d.step()
            disc_loss.append(loss_discri.item())

            fake_feat_d = discriminator(fake_imgs)

            logits_gen = {
                "fake": {
                    "word_level": discriminator.logits_word_level(
                        fake_feat_d, word_emb
                    ),
                    "uncond": discriminator.logits_uncond(fake_feat_d),
                    "cond": discriminator.logits_cond(fake_feat_d, sent_emb),
                }
            }

            # Update Generator
            optimizer_g.zero_grad()
            loss_gen = generator_loss(
                logits_gen,
                local_fake_incept_feat,
                global_fake_incept_feat,
                labels_real,
                word_labels,
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
            optimizer_text_encoder.zero_grad()
            optimizer_text_encoder.step()

            # Update the moving average of the generator parameters
            for param, avg_p in zip(generator.parameters(), g_param_avg):
                avg_p = smooth_val_gen * avg_p + (1 - smooth_val_gen) * param.data

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}],\
                    Loss D: {loss_discri.item():.4f}, Loss G: {loss_gen.item():.4f}"
                )

            if (batch_idx + 1) % 50 == 0:
                with torch.no_grad():
                    g_backup_params = copy_gen_params(generator)
                    load_params(generator, g_param_avg)
                    fake_imgs, _, _ = generator(
                        noise,
                        sent_emb,
                        word_emb,
                        global_incept_feat,
                        local_incept_feat,
                        vgg_feat,
                        mask,
                    )
                    save_image_and_caption(
                        fake_imgs,
                        images,
                        correct_capt,
                        ix2word,
                        batch_idx,
                        epoch,
                        output_dir,
                    )
                    load_params(generator, g_backup_params)
                    save_plot(gen_loss, disc_loss, epoch, batch_idx, output_dir)

        if epoch % snapshot == 0 and epoch != 0:
            save_model(generator, discriminator, g_param_avg, epoch, output_dir)

    save_model(generator, discriminator, g_param_avg, epochs, output_dir)
