"""Code to Initiate training of the model."""

import argparse
import torch
import nltk
from torch.utils.data import DataLoader
from src.models.train_model import train
from src.data.datasets import TextImageDataset
from src.data.collate import custom_collate
from torchvision import transforms
from src.config import config_dict, update_config
from functools import partial

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def main(args):
    """Main function to initiate training."""
    cfg_dict = config_dict
    # Define dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.Resize(304),
        ]
    )
    dataset = TextImageDataset(args.data_dir, args.split, int(args.num_capt), transform)
    ix2word = dataset.ix_to_word
    vocab_len = len(ix2word)
    batch_size = int(args.batch_size)
    cfg_dict = update_config(cfg_dict, ix2word = ix2word, vocab_len = vocab_len, batch_size = batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last = True,
        shuffle=True,
        num_workers=args.num_workers,
        multiprocessing_context = "spawn",
        collate_fn=partial(custom_collate, device = cfg_dict["device"]),
    )

    # Train the model
    train(dataloader, cfg_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
        required = True
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of dataset to use for training. Can be 'train' or 'test'.",
        required = True
    )
    parser.add_argument(
        "--num_capt",
        type=int,
        default=5,
        help="Number of captions per image present in the dataset. 5 for COCO, 10 for bird.",
        required = True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    args = parser.parse_args()
    main(args)
