import pickle
import re
from typing import List


class TAIMGANTokenizer:
    def __init__(self, captions_path):
        with open(captions_path, "rb") as ckpt_file:
            captions = pickle.load(ckpt_file)
            self.ix_to_word = captions[2]
            self.word_to_ix = captions[3]
        self.token_regex = r'\w+'
        self.pad_token_id = self.word_to_ix["<end>"]
        self.pad_repr = "[PAD]"

    def encode(self, text: str) -> List[int]:
        return [self.word_to_ix.get(word, self.pad_token_id)
                for word in re.findall(self.token_regex, text.lower())]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.ix_to_word[token]
                         if token != self.pad_token_id else self.pad_repr
                         for token in tokens])
