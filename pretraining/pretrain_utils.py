import os
import pickle
import torch
from torch.utils.data.dataset import Dataset 
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, List, Optional
import time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

class LineByLineTextDatasetCached(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_lbl_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )
        if os.path.exists(cached_features_file):
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
            self.examples = batch_encoding["input_ids"]
            self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
