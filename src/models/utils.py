"""Helper functions for models."""

import pathlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import optim

from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import Generator

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def copy_gen_params(generator: Generator) -> Any:
    """
    Function to copy the parameters of the generator
    """
    params = deepcopy(list(p.data for p in generator.parameters()))
    return params


def define_optimizers(
    generator: Generator, discriminator: Discriminator, disc_lr: float, gen_lr: float
) -> Any:
    """
    Function to define the optimizers for the generator and discriminator
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param disc_lr: Learning rate for the discriminator
    :param gen_lr: Learning rate for the generator
    """
    optimizer_g = optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    return optimizer_g, optimizer_d


def prepare_labels(batch_size: int, max_seq_len: int, device: torch.device) -> Any:
    """
    Function to prepare the labels for the discriminator and generator.
    """
    real_labels = torch.FloatTensor(batch_size, 1).fill_(1).to(device)
    fake_labels = torch.FloatTensor(batch_size, 1).fill_(0).to(device)
    match_labels = torch.LongTensor(range(batch_size)).to(device)
    fake_word_labels = torch.FloatTensor(batch_size, max_seq_len).fill_(0).to(device)

    return real_labels, fake_labels, match_labels, fake_word_labels


def load_params(generator: Generator, new_params: Any) -> Any:
    """
    Function to load new parameters to the generator
    """
    for param, new_p in zip(generator.parameters(), new_params):
        param.data.copy_(new_p)


def get_image_arr(image_tensor: torch.Tensor) -> Any:
    """
    Function to convert a tensor to an image array.
    :param image_tensor: Tensor containing the image (shape: (batch_size, channels, height, width))
    """

    image = image_tensor.cpu().detach().numpy()
    image = (image + 1) * (255 / 2.0)
    image = np.transpose(image, (0, 2, 3, 1))  # (B,C,H,W) -> (B,H,W,C)
    image = image.astype(np.uint8)
    return image  # (B,H,W,C)


def get_captions(captions: torch.Tensor, ix2word: dict[int, str]) -> Any:
    """
    Function to convert a tensor to a list of captions.
    :param captions: Tensor containing the captions (shape: (batch_size, max_seq_len))
    :param ix2word: Dictionary mapping indices to words
    """
    captions = captions.cpu().detach().numpy()
    captions = [[ix2word[ix] for ix in cap if ix != 0] for cap in captions]  # type: ignore
    return captions


def save_model(
    generator: Generator,
    discriminator: Discriminator,
    params: Any,
    epoch: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save the model.
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param params: Parameters of the generator
    :param epoch: Epoch number
    :param output_dir: Output directory
    """
    output_path = output_dir / "weights/"
    backup_para = copy_gen_params(generator)
    load_params(generator, params)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), output_path / f"generator_epoch_{epoch}.pth")
    load_params(generator, backup_para)
    torch.save(
        discriminator.state_dict(), output_path / f"discriminator_epoch_{epoch}.pth"
    )
    print(f"Model saved at epoch {epoch}.")


def save_image_and_caption(
    fake_img_tensor: torch.Tensor,
    img_tensor: torch.Tensor,
    captions: torch.Tensor,
    ix2word: dict[int, str],
    batch_idx: int,
    epoch: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save an image and its corresponding caption.
    :param fake_img_tensor: Tensor containing the generated image
    (shape: (batch_size, channels, height, width))

    :param img_tensor: Tensor containing the image
    (shape: (batch_size, channels, height, width))

    :param captions: Tensor containing the captions
    (shape: (batch_size, max_seq_len))

    :param ix2word: Dictionary mapping indices to words
    :param batch_idx: Batch index
    :param epoch: Epoch number
    :param output_dir: Output directory
    """
    output_path = output_dir
    output_path_text = output_dir
    capt_list = get_captions(captions, ix2word)
    img_arr = get_image_arr(img_tensor)
    fake_img_arr = get_image_arr(fake_img_tensor)

    for i in range(img_arr.shape[0]):
        img = Image.fromarray(img_arr[i])
        fake_img = Image.fromarray(fake_img_arr[i])

        fake_img_path = (
            output_path / f"generated/{epoch}_epochs/{batch_idx}_batch/{i+1}.png"
        )
        img_path = output_path / f"real/{epoch}_epochs/{batch_idx}_batch/{i+1}.png"
        text_path = (
            output_path_text / f"text/{epoch}_epochs/{batch_idx}_batch/captions.txt"
        )

        Path(fake_img_path).parent.mkdir(parents=True, exist_ok=True)
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
        Path(text_path).parent.mkdir(parents=True, exist_ok=True)

        fake_img.save(fake_img_path)
        img.save(img_path)

        with open(text_path, "a", encoding="utf-8") as txt_file:
            text_str = str(i + 1) + ": " + " ".join(capt_list[i])
            txt_file.write(text_str)
            txt_file.write("\n")
