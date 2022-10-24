"""Pytorch Dataset classes for the datasets used in the project."""

# Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import Any
import pickle


class TextImageDataset(Dataset):
    """Custom PyTorch Dataset class to load Image and Text data."""

    def __init__(self, data_path: str, split: str, transform: Any=None):
        """
        :param data_path: Path to the data directory. [i.e. can be './data/birds/', or './data/coco/]
        :param split: 'train' or 'test' split
        :param transform: PyTorch transform to apply to the images.
        """
        self.transform = transform
        self.bound_box_map = None
        self.file_names = self.load_filenames(data_path, split)
        self.data_path = data_path
        if self.data_path  == './birds/':
            self.bound_box_map = self.get_bound_box(data_path)

        elif self.data_path == './coco/':
            pass

        else:
            raise ValueError("Invalid data path. Please use './data/birds/' or './data/coco/'")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.file_names)
    
    def __getitem__(self, idx):
        """Return the item at index idx."""
        key = self.file_names[idx]

        if self.bound_box_map is not None:
            bbox = self.bound_box_map[key]
            images_dir = os.path.join(self.data_path, 'CUB_200_2011/images')
        else:
            bbox = None
            images_dir = os.path.join(self.data_path, 'images')

        img_path = os.path.join(images_dir, key + '.jpg')
        img_tensor = self.get_image(img_path, bbox, self.transform)

        return img_tensor

    def get_image(self, img_path: str, bbox: list, transform: Any):
        """
        Helper function to load and transform an image.
        :param img_path: path to the image
        :param bbox: bounding box coordinates [x, y, width, height]
        :param transform: PyTorch transform to apply to the image
        :return img_tensor: transformed image tensor
        """
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75) 

            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

            img = img.crop([x1, y1, x2, y2]) #This preprocessing steps seems to follow from Stackgan: Text to photo-realistic image synthesis

        if transform is not None:
            img_tensor = transform(img) # this scales to 304x304, i.e. 256 x (76/64). 
            x = np.random.randint(0, 48) # 304 - 256 = 48
            y = np.random.randint(0, 48)
            flip = np.random.rand() > 0.5

            #crop
            img_tensor = img_tensor.crop([x, y, x + 256, y + 256]) # this crops to 256x256
            if flip:
                img_tensor = F.hflip(img_tensor)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        img_tensor = normalize(img_tensor)

        return img_tensor

    def load_filenames(self, data_dir: str, split: str):
        """
        Helper function to get list of all image filenames.
        :param data_dir: path to the data directory [i.e. './data/birds/' or './data/coco/']
        :param split: 'train' or 'test' split
        :return filenames: list of all image filenames
        """
        filepath = f"{data_dir}{split}/filenames.pickle"
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
        else:
            filenames = []
            raise ValueError("Invalid split. Please use 'train' or 'test', or make sure the filenames.pickle file exists.")
        return filenames
    
    
    def get_bound_box(self, data_path):
        """
        Helper function to get the bounding box for birds dataset.
        :param data_path: path to birds data directory [i.e. './data/birds/']
        :return imageToBox: dictionary mapping image name to bounding box coordinates
        """
        bbox_path = os.path.join(data_path, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_path, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist() # df_filenames[0] just contains the index or ID.

        imageToBox = {img_file[:-4]: [] for img_file in filenames} # remove the .jpg extension from the names
        numImgs = len(filenames)

        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            imageToBox[key] = bbox

        return imageToBox
