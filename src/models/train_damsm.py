"""Module to train the GAN model"""

from typing import Any

import torch

from src.models.losses import generator_loss_damsm
from src.models.modules.image_encoder import InceptionEncoder
from src.models.modules.text_encoder import TextEncoder
from src.models.utils import (
    define_optimizers_damsm,
    prepare_labels,
    save_model_damsm,
    save_plot_damsm,
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

    text_encoder = TextEncoder(vocab_len, D, D // 2).to(device)
    image_encoder = InceptionEncoder(D).to(device)
    gen_loss = []

    (
        optimizer_text_encoder,
        opt_image_encoder,
    ) = define_optimizers_damsm(
        image_encoder, text_encoder, lr_config
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

            optimizer_text_encoder.zero_grad()
            opt_image_encoder.zero_grad()

            word_emb, sent_emb = text_encoder(correct_capt)

            local_incept_feat, global_incept_feat = image_encoder(images)

            # Update Generator
            loss_gen = generator_loss_damsm(
                labels_real,
                word_emb,
                sent_emb,
                labels_match,
                correct_capt_len,
                curr_class,
                local_incept_feat,
                global_incept_feat,
                const_dict,
            )

            loss_gen.backward()
            optimizer_text_encoder.step()
            opt_image_encoder.step()
            gen_loss.append(loss_gen.item())

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}],\
                    Loss DAMSM: {loss_gen.item():.4f}"
                )

            if (batch_idx + 1) % 50 == 0:
                with torch.no_grad():
                    save_plot_damsm(gen_loss, epoch, batch_idx, output_dir)

        if epoch % snapshot == 0 and epoch != 0:
            save_model_damsm(
                image_encoder, text_encoder, epoch, output_dir
            )

    save_model_damsm(
        image_encoder, text_encoder, epochs, output_dir
    )
