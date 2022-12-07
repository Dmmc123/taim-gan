"""Helper functions for models."""

import pathlib
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import optim

from src.models.modules.discriminator import Discriminator
from src.models.modules.generator import Generator
from src.models.modules.image_encoder import InceptionEncoder
from src.models.modules.text_encoder import TextEncoder

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def copy_gen_params(generator: Generator) -> Any:
    """
    Function to copy the parameters of the generator
    """
    params = deepcopy(list(p.data for p in generator.parameters()))
    return params


def define_optimizers_damsm(
    image_encoder: InceptionEncoder,
    text_encoder: TextEncoder,
    lr_config: dict[str, float],
) -> Any:
    """
    Function to define the optimizers for the generator and discriminator
    :param generator: Generator model
    :param image_encoder: Image encoder model
    :param text_encoder: Text encoder model
    :param discriminator: Discriminator model
    :param lr_config: Dictionary containing the learning rates for the optimizers

    """
    img_encoder_lr = lr_config["img_encoder_lr"]
    text_encoder_lr = lr_config["text_encoder_lr"]

    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=text_encoder_lr)
    optimizer_image_encoder = optim.Adam(image_encoder.parameters(), lr=img_encoder_lr)

    return optimizer_text_encoder, optimizer_image_encoder


def define_optimizers(
    generator: Generator,
    discriminator: Discriminator,
    image_encoder: InceptionEncoder,
    text_encoder: TextEncoder,
    lr_config: Dict[str, float],
) -> Any:
    """
    Function to define the optimizers for the generator and discriminator
    :param generator: Generator model
    :param image_encoder: Image encoder model
    :param text_encoder: Text encoder model
    :param discriminator: Discriminator model
    :param lr_config: Dictionary containing the learning rates for the optimizers

    """
    img_encoder_lr = lr_config["img_encoder_lr"]
    text_encoder_lr = lr_config["text_encoder_lr"]
    gen_lr = lr_config["gen_lr"]
    disc_lr = lr_config["disc_lr"]

    optimizer_g = optim.Adam(
        [{"params": generator.parameters()}],
        lr=gen_lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        [{"params": discriminator.parameters()}],
        lr=disc_lr,
        betas=(0.5, 0.999),
    )
    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=text_encoder_lr)
    optimizer_image_encoder = optim.Adam(image_encoder.parameters(), lr=img_encoder_lr)

    return optimizer_g, optimizer_d, optimizer_text_encoder, optimizer_image_encoder


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


def get_captions(captions: torch.Tensor, ix2word: Dict[int, str]) -> Any:
    """
    Function to convert a tensor to a list of captions.
    :param captions: Tensor containing the captions (shape: (batch_size, max_seq_len))
    :param ix2word: Dictionary mapping indices to words
    """
    captions = captions.cpu().detach().numpy()
    captions = [[ix2word[ix] for ix in cap if ix != 0] for cap in captions]  # type: ignore
    return captions


def save_model_damsm(
    image_encoder: InceptionEncoder,
    text_encoder: TextEncoder,
    epoch: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save the model.
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param image_encoder: Image encoder model
    :param text_encoder: Text encoder model
    :param params: Parameters of the generator
    :param epoch: Epoch number
    :param output_dir: Output directory
    """
    output_path = output_dir / "weights_damsm/"
    Path(output_path / "image_encoder").mkdir(parents=True, exist_ok=True)
    torch.save(
        image_encoder.state_dict(),
        output_path / f"image_encoder/image_encoder_epoch_{epoch}.pth",
    )
    Path(output_path / "text_encoder").mkdir(parents=True, exist_ok=True)
    torch.save(
        text_encoder.state_dict(),
        output_path / f"text_encoder/text_encoder_epoch_{epoch}.pth",
    )
    print(f"Model saved at epoch {epoch}.")


def save_model(
    generator: Generator,
    discriminator: Discriminator,
    image_encoder: InceptionEncoder,
    text_encoder: TextEncoder,
    epoch: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save the model.
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param image_encoder: Image encoder model
    :param text_encoder: Text encoder model
    :param params: Parameters of the generator
    :param epoch: Epoch number
    :param output_dir: Output directory
    """
    output_path = output_dir / "weights/"
    Path(output_path / "generator").mkdir(parents=True, exist_ok=True)
    torch.save(
        generator.state_dict(), output_path / f"generator/generator_epoch_{epoch}.pth"
    )
    Path(output_path / "discriminator").mkdir(parents=True, exist_ok=True)
    torch.save(
        discriminator.state_dict(),
        output_path / f"discriminator/discriminator_epoch_{epoch}.pth",
    )
    Path(output_path / "image_encoder").mkdir(parents=True, exist_ok=True)
    torch.save(
        image_encoder.state_dict(),
        output_path / f"image_encoder/image_encoder_epoch_{epoch}.pth",
    )
    Path(output_path / "text_encoder").mkdir(parents=True, exist_ok=True)
    torch.save(
        text_encoder.state_dict(),
        output_path / f"text_encoder/text_encoder_epoch_{epoch}.pth",
    )
    print(f"Model saved at epoch {epoch}.")


def save_image_and_caption(
    fake_img_tensor: torch.Tensor,
    img_tensor: torch.Tensor,
    captions: torch.Tensor,
    ix2word: Dict[int, str],
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

def save_plot_damsm(
    gen_loss: list[float],
    epoch: int,
    batch_idx: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save the plot of the loss.
    :param gen_loss: List of generator losses
    :param disc_loss: List of discriminator losses
    :param epoch: Epoch number
    :param batch_idx: Batch index
    :param output_dir: Output directory
    """
    pickle_path = output_dir / "losses_damsm/"
    output_path = output_dir / "plots_damsm" / f"{epoch}_epochs/{batch_idx}_batch/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(pickle_path).mkdir(parents=True, exist_ok=True)

    with open(pickle_path / "gen_loss.pkl", "wb") as pickl_file:
        pickle.dump(gen_loss, pickl_file)

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(24, 12))
    plt.plot(gen_loss, label="DAMSM Loss")
    plt.xlabel("No of Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path / "loss.png", bbox_inches="tight")
    plt.clf()
    plt.close()



def save_plot(
    gen_loss: List[float],
    disc_loss: List[float],
    epoch: int,
    batch_idx: int,
    output_dir: pathlib.PosixPath,
) -> None:
    """
    Function to save the plot of the loss.
    :param gen_loss: List of generator losses
    :param disc_loss: List of discriminator losses
    :param epoch: Epoch number
    :param batch_idx: Batch index
    :param output_dir: Output directory
    """
    pickle_path = output_dir / "losses/"
    output_path = output_dir / "plots" / f"{epoch}_epochs/{batch_idx}_batch/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(pickle_path).mkdir(parents=True, exist_ok=True)

    with open(pickle_path / "gen_loss.pkl", "wb") as pickl_file:
        pickle.dump(gen_loss, pickl_file)

    with open(pickle_path / "disc_loss.pkl", "wb") as pickl_file:
        pickle.dump(disc_loss, pickl_file)

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(24, 12))
    plt.plot(gen_loss, label="Generator Loss")
    plt.plot(disc_loss, label="Discriminator Loss")
    plt.xlabel("No of Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path / "loss.png", bbox_inches="tight")
    plt.clf()
    plt.close()


def load_model(
    generator: Generator,
    discriminator: Discriminator,
    image_encoder: InceptionEncoder,
    text_encoder: TextEncoder,
    output_dir: pathlib.Path,
    device: torch.device
) -> None:
    """
    Function to load the model.
    :param generator: Generator model
    :param discriminator: Discriminator model
    :param image_encoder: Image encoder model
    :param text_encoder: Text encoder model
    :param output_dir: Output directory
    :param device: device to map the location of weights
    """
    if (output_dir / "generator.pth").exists():
        generator.load_state_dict(torch.load(output_dir / "generator.pth", map_location=device))
        print("Generator loaded.")
    if (output_dir / "discriminator.pth").exists():
        discriminator.load_state_dict(torch.load(output_dir / "discriminator.pth", map_location=device))
        print("Discriminator loaded.")
    if (output_dir / "image_encoder.pth").exists():
        image_encoder.load_state_dict(torch.load(output_dir / "image_encoder.pth", map_location=device))
        print("Image Encoder loaded.")
    elif (output_path / "image_encoder_damsm_utkface.pth").exists():
        image_encoder.load_state_dict(torch.load(output_path / "image_encoder_damsm_utkface.pth"))
        print("UTKFace DAMSM Image Encoder loaded.")
    if (output_path / "text_encoder.pth").exists():
        text_encoder.load_state_dict(torch.load(output_path / "text_encoder.pth"))
        print("Text Encoder loaded.")
    elif (output_path / "text_encoder_damsm_utkface.pth").exists():
        text_encoder.load_state_dict(torch.load(output_path / "text_encoder_damsm_utkface.pth"))
        print("UTKFace DAMSM Text Encoder loaded.")
