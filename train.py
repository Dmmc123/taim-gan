"""Code to Initiate training of the model."""

import argparse
from torch.utils.data import DataLoader
from src.models.train_model import train
from src.data.datasets import TextImageDataset
from src.data.collate import custom_collate
from torchvision import transforms
from src.config import config_dict, update_config


def main(args):
    """Main function to initiate training."""
    config_dict = config_dict
    # Define dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.Resize(304),
        ]
    )
    dataset = TextImageDataset(args.data_dir, args.split, args.num_capt, transform)
    ix2word = dataset.ix_to_word
    vocab_len = len(ix2word)
    config_dict = update_config(config_dict, ix2word = ix2word, vocab_len = vocab_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
    )

    # Train the model
    train(dataloader, config_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of dataset to use for training. Can be 'train' or 'test'.",
    )
    parser.add_argument(
        "--num_capt",
        type=int,
        default=5,
        help="Number of captions per image present in the dataset. 5 for COCO, 10 for bird.",
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
