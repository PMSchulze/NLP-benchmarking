import argparse
from datasets import load_dataset, load_metric
import itertools
import json
import numpy as np
import os
import os.path
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import utils_gpt2_glue

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--cache_dir")
parser.add_argument("--model_name_or_path")
parser.add_argument("--num_train_epochs", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--seed", type=int)
parser.add_argument("--token_vocab")
args = parser.parse_args()

# Setup CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load training and evaluation data
data_train = load_dataset(
    'glue', 
    'mnli', 
    split = 'train',
    cache_dir = args.cache_dir
)
data_eval_matched = load_dataset(
    'glue', 
    'mnli_matched', 
    split = 'validation', 
    cache_dir = args.cache_dir
)
data_eval_mismatched = load_dataset(
    'glue', 
    'mnli_mismatched', 
    split = 'validation', 
    cache_dir = args.cache_dir
)

# Load GPT2 tokenizer
utils_gpt2_glue.tokenizer = GPT2Tokenizer.from_pretrained(
    args.token_vocab, 
    bos_token = '<|start|>', 
    eos_token = '<|end|>', 
    pad_token = '<pad>',
)

# Add special tokens to tokenizer
utils_gpt2_glue.tokenizer.add_tokens(["<$>"])

# Rename column 'label' to 'labels', which is the expected keyword argument 
# of huggingface's models.
data_train.rename_column_('label', 'labels')
data_eval_matched.rename_column_('label', 'labels')
data_eval_mismatched.rename_column_('label', 'labels')

# Specify columns that do not contain of input for the model and can thus be
# dropped. 
remove_cols = ['idx', 'premise', 'hypothesis']

# Tokenize the sentences dependent on the task, using the same method that was 
# used in the original GPT
data_train = data_train.map(lambda x: utils_gpt2_glue.encode(x, 'MNLI'), 
                            batched = True, remove_columns = remove_cols, 
                            keep_in_memory = True)
data_eval_matched = data_eval_matched.map(lambda x: utils_gpt2_glue.encode(x, 'MNLI'), 
                          batched = True, remove_columns = remove_cols, 
                          keep_in_memory = True)
data_eval_mismatched = data_eval_mismatched.map(lambda x: utils_gpt2_glue.encode(x, 'MNLI'), 
                          batched = True, remove_columns = remove_cols, 
                          keep_in_memory = True)
