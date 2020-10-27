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

# Convert data to torch.tensor
data_train.set_format(type = 'torch')
data_eval_matched.set_format(type = 'torch')
data_eval_mismatched.set_format(type = 'torch')

# Set seed before creating and shuffling the batches for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create input batches; the training data is randomly shuffled to implement SGD
batches_train = DataLoader(data_train, sampler = RandomSampler(data_train),
                           batch_size = args.batch_size)
batches_eval_matched = DataLoader(data_eval_matched, sampler = SequentialSampler(data_eval), 
                          batch_size = args.batch_size)
batches_eval_mismatched = DataLoader(data_eval_mismatched, sampler = SequentialSampler(data_eval), 
                          batch_size = args.batch_size)

# Drop the cached data (only the batches are needed)
data_train.cleanup_cache_files()
data_eval_matched.cleanup_cache_files()
data_eval_mismatched.cleanup_cache_files()

# Instatiate the model:
model = utils_gpt2_glue.GPT2ForSequenceClassification(
    n_classes = n_classes,
    gpt_model_name_or_path = args.model_name_or_path,
)

# Add new tokens (<start>, <end>) to the embedding matrix
# Weights are randomly initialized, as in GPT paper
model.gpt2model.resize_token_embeddings(len(utils_gpt2_glue.tokenizer))

# Specify optimizer and hyperparameters
optimizer = AdamW(
    model.parameters(),
    lr = 2e-5,
    eps = 1e-8
)

# Calculate number of training steps
total_steps = len(batches_train) * args.num_train_epochs

# Create the learning rate scheduler (with num_warmup_steps=0, as in other models fine-tuned on GLUE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

# Activate CUDA
model.cuda()

