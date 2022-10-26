"""Pytorch Dataset classes for the datasets used in the project."""

import os
import pickle
from collections import defaultdict
from typing import Any

import nltk
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TextImageDataset(Dataset):  # type: ignore
    """Custom PyTorch Dataset class to load Image and Text data."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-function-args

    def __init__(
        self, data_path: str, split: str, num_captions: int, transform: Any = None
    ):
        """
        :param data_path: Path to the data directory. [i.e. can be './birds/', or './coco/]
        :param split: 'train' or 'test' split
        :param num_captions: number of captions present per image.
        [For birds, this is 10, for coco, this is 5]
        :param transform: PyTorch transform to apply to the images.
        """
        self.transform = transform
        self.bound_box_map = None
        self.file_names = self.load_filenames(data_path, split)
        self.data_path = data_path
        self.num_captions_per_image = num_captions
        (
            self.captions,
            self.ix_to_word,
            self.word_to_ix,
            self.vocab_len,
        ) = self.get_capt_and_vocab(data_path, split)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.class_ids = self.get_class_id(data_path, split, len(self.file_names))
        if self.data_path == "./birds/":
            self.bound_box_map = self.get_bound_box(data_path)

        elif self.data_path == "./coco/":
            pass

        else:
            raise ValueError(
                "Invalid data path. Please use './data/birds/' or './data/coco/'"
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Any:
        """
        Return the item at index idx.
        :param idx: index of the item to return
        :return img_tensor: image tensor
        :return correct_caption: correct caption for the image [list of word indices]
        :return curr_class_id: class id of the image
        :return wrong_caption: wrong caption for the image [list of word indices]
        :return wrong_class_id: class id of the wrong caption
        :return word_labels: POS_tagged word labels [1 for noun and adjective, 0 else]

        """
        file_name = self.file_names[idx]
        curr_class_id = self.class_ids[idx]

        if self.bound_box_map is not None:
            bbox = self.bound_box_map[file_name]
            images_dir = os.path.join(self.data_path, "CUB_200_2011/images")
        else:
            bbox = None
            images_dir = os.path.join(self.data_path, "images")

        img_path = os.path.join(images_dir, file_name + ".jpg")
        img_tensor = self.get_image(img_path, bbox, self.transform)

        rand_sent_idx = np.random.randint(0, self.num_captions_per_image)
        rand_sent_idx = idx * self.num_captions_per_image + rand_sent_idx

        correct_caption = torch.tensor(self.captions[rand_sent_idx], dtype=torch.int64)
        num_words = len(correct_caption)

        wrong_idx = np.random.randint(0, len(self.file_names))
        wrong_class_id = self.class_ids[wrong_idx]
        while wrong_class_id == curr_class_id:
            wrong_idx = np.random.randint(0, len(self.file_names))
            wrong_class_id = self.class_ids[wrong_idx]

        wrong_sent_idx = np.random.randint(0, self.num_captions_per_image)
        wrong_sent_idx = wrong_idx * self.num_captions_per_image + wrong_sent_idx

        wrong_caption = torch.tensor(self.captions[wrong_sent_idx], dtype=torch.int64)

        capt_token_list = []
        for i in range(num_words):
            capt_token_list.append(self.ix_to_word[correct_caption[i].item()])

        pos_tag_list = nltk.tag.pos_tag(capt_token_list)
        word_labels = []

        for pos_tag in pos_tag_list:
            if (
                "NN" in pos_tag[1] or "JJ" in pos_tag[1]
            ):  # check for Nouns and Adjective only
                word_labels.append(1)
            else:
                word_labels.append(0)

        word_labels = torch.tensor(word_labels, dtype=torch.int64)  # type: ignore

        curr_class_id = torch.tensor(curr_class_id, dtype=torch.int64).unsqueeze(0)
        wrong_class_id = torch.tensor(wrong_class_id, dtype=torch.int64).unsqueeze(0)

        return (
            img_tensor,
            correct_caption,
            curr_class_id,
            wrong_caption,
            wrong_class_id,
            word_labels,
        )

    def get_capt_and_vocab(self, data_dir: str, split: str) -> Any:
        """
        Helper function to get the captions, vocab dict for each image.
        :param data_dir: path to the data directory [i.e. './birds/' or './coco/']
        :param split: 'train' or 'test' split
        :return captions: list of all captions for each image
        :return ix_to_word: dictionary mapping index to word
        :return word_to_ix: dictionary mapping word to index
        :return num_words: number of unique words in the vocabulary
        """
        captions_ckpt_path = os.path.join(data_dir, "captions.pickle")
        if os.path.exists(
            captions_ckpt_path
        ):  # check if previously processed captions exist
            with open(captions_ckpt_path, "rb") as ckpt_file:
                captions = pickle.load(ckpt_file)
                train_captions, test_captions = captions[0], captions[1]
                ix_to_word, word_to_ix = captions[2], captions[3]
                num_words = len(ix_to_word)
                del captions
                if split == "train":
                    return train_captions, ix_to_word, word_to_ix, num_words
                return test_captions, ix_to_word, word_to_ix, num_words

        else:  # if not, process the captions and save them
            train_files = self.load_filenames(data_dir, "train")
            test_files = self.load_filenames(data_dir, "test")

            train_captions_tokenized = self.get_tokenized_captions(
                data_dir, train_files
            )
            test_captions_tokenized = self.get_tokenized_captions(
                data_dir, test_files
            )  # we need both train and test captions to build the vocab

            (
                train_captions,
                test_captions,
                ix_to_word,
                word_to_ix,
                num_words,
            ) = self.build_vocab(  # type: ignore
                train_captions_tokenized, test_captions_tokenized, split
            )
            vocab_list = [train_captions, test_captions, ix_to_word, word_to_ix]
            with open(captions_ckpt_path, "wb") as ckpt_file:
                pickle.dump(vocab_list, ckpt_file)

            if split == "train":
                return train_captions, ix_to_word, word_to_ix, num_words
            if split == "test":
                return test_captions, ix_to_word, word_to_ix, num_words
            raise ValueError("Invalid split. Please use 'train' or 'test'")

    def build_vocab(
        self, tokenized_captions_train: list, tokenized_captions_test: list  # type: ignore
    ) -> Any:
        """
        Helper function which builds the vocab dicts.
        :param tokenized_captions_train: list containing all the
        train tokenized captions in the dataset. This is list of lists.
        :param tokenized_captions_test: list containing all the
        test tokenized captions in the dataset. This is list of lists.
        :return train_captions_int: list of all captions in training,
        where each word is replaced by its index in the vocab
        :return test_captions_int: list of all captions in test,
        where each word is replaced by its index in the vocab
        :return ix_to_word: dictionary mapping index to word
        :return word_to_ix: dictionary mapping word to index
        :return num_words: number of unique words in the vocabulary
        """
        vocab = defaultdict(int)  # type: ignore
        total_captions = tokenized_captions_train + tokenized_captions_test
        for caption in total_captions:
            for word in caption:
                vocab[word] += 1

        # sort vocab dict by frequency in descending order
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)  # type: ignore

        ix_to_word = {}
        word_to_ix = {}
        ix_to_word[0] = "<end>"
        word_to_ix["<end>"] = 0

        word_idx = 1
        for word, _ in vocab:
            word_to_ix[word] = word_idx
            ix_to_word[word_idx] = word
            word_idx += 1

        train_captions_int = []  # we want to convert words to indices in vocab.
        for caption in tokenized_captions_train:
            curr_caption_int = []
            for word in caption:
                curr_caption_int.append(word_to_ix[word])

            train_captions_int.append(curr_caption_int)

        test_captions_int = []
        for caption in tokenized_captions_test:
            curr_caption_int = []
            for word in caption:
                curr_caption_int.append(word_to_ix[word])

            test_captions_int.append(curr_caption_int)

        return (
            train_captions_int,
            test_captions_int,
            ix_to_word,
            word_to_ix,
            len(ix_to_word),
        )

    def get_tokenized_captions(self, data_dir: str, filenames: list) -> Any:  # type: ignore
        """
        Helper function to tokenize and return captions for each image in filenames.
        :param data_dir: path to the data directory [i.e. './birds/' or './coco/']
        :param filenames: list of all filenames corresponding to the split
        :return tokenized_captions: list of all tokenized captions for all files in filenames.
        [this returns a list, where each element is again a list of tokens/words]
        """

        all_captions = []
        for filename in filenames:
            caption_path = os.path.join(data_dir, "text", filename + ".txt")
            with open(caption_path, "r", encoding="utf8") as txt_file:
                captions = txt_file.readlines()
                count = 0
                for caption in captions:
                    if len(caption) == 0:
                        continue

                    caption = caption.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(
                        caption.lower()
                    )  # splits current caption/line to list of words/tokens
                    if len(tokens) == 0:
                        continue

                    tokens = [
                        t.encode("ascii", "ignore").decode("ascii") for t in tokens
                    ]
                    tokens = [t for t in tokens if len(t) > 0]

                    all_captions.append(tokens)
                    count += 1
                    if count == self.num_captions_per_image:
                        break
                    if count < self.num_captions_per_image:
                        raise ValueError(
                            f"Number of captions for {filename} is only {count},\
                                which is less than {self.num_captions_per_image}."
                        )

        return all_captions

    def get_image(self, img_path: str, bbox: list, transform: Any) -> Any:  # type: ignore
        """
        Helper function to load and transform an image.
        :param img_path: path to the image
        :param bbox: bounding box coordinates [x, y, width, height]
        :param transform: PyTorch transform to apply to the image
        :return img_tensor: transformed image tensor
        """
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        if bbox is not None:
            r_val = int(np.maximum(bbox[2], bbox[3]) * 0.75)

            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1_coord = np.maximum(0, center_y - r_val)
            y2_coord = np.minimum(height, center_y + r_val)
            x1_coord = np.maximum(0, center_x - r_val)
            x2_coord = np.minimum(width, center_x + r_val)

            img = img.crop(
                [x1_coord, y1_coord, x2_coord, y2_coord]
            )  # This preprocessing steps seems to follow from
            # Stackgan: Text to photo-realistic image synthesis

        if transform is not None:
            img_tensor = transform(img)  # this scales to 304x304, i.e. 256 x (76/64).
            x_val = np.random.randint(0, 48)  # 304 - 256 = 48
            y_val = np.random.randint(0, 48)
            flip = np.random.rand() > 0.5

            # crop
            img_tensor = img_tensor.crop(
                [x_val, y_val, x_val + 256, y_val + 256]
            )  # this crops to 256x256
            if flip:
                img_tensor = F.hflip(img_tensor)

        img_tensor = self.normalize(img_tensor)

        return img_tensor

    def load_filenames(self, data_dir: str, split: str) -> Any:
        """
        Helper function to get list of all image filenames.
        :param data_dir: path to the data directory [i.e. './birds/' or './coco/']
        :param split: 'train' or 'test' split
        :return filenames: list of all image filenames
        """
        filepath = f"{data_dir}{split}/filenames.pickle"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as pick_file:
                filenames = pickle.load(pick_file)
        else:
            raise ValueError(
                "Invalid split. Please use 'train' or 'test',\
                     or make sure the filenames.pickle file exists."
            )
        return filenames

    def get_class_id(self, data_dir: str, split: str, total_elems: int) -> Any:
        """
        Helper function to get list of all image class ids.
        :param data_dir: path to the data directory [i.e. './birds/' or './coco/']
        :param split: 'train' or 'test' split
        :param total_elems: total number of elements in the dataset
        :return class_ids: list of all image class ids
        """
        filepath = f"{data_dir}{split}/class_info.pickle"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as class_file:
                class_ids = pickle.load(class_file, encoding="latin1")
        else:
            class_ids = np.arange(total_elems)
        return class_ids

    def get_bound_box(self, data_path: str) -> Any:
        """
        Helper function to get the bounding box for birds dataset.
        :param data_path: path to birds data directory [i.e. './data/birds/']
        :return imageToBox: dictionary mapping image name to bounding box coordinates
        """
        bbox_path = os.path.join(data_path, "CUB_200_2011/bounding_boxes.txt")
        df_bounding_boxes = pd.read_csv(
            bbox_path, delim_whitespace=True, header=None
        ).astype(int)

        filepath = os.path.join(data_path, "CUB_200_2011/images.txt")
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[
            1
        ].tolist()  # df_filenames[0] just contains the index or ID.

        img_to_box = {  # type: ignore
            img_file[:-4]: [] for img_file in filenames
        }  # remove the .jpg extension from the names
        num_imgs = len(filenames)

        for i in range(0, num_imgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            img_to_box[key] = bbox

        return img_to_box
