import argparse
from datasets import load_dataset
import glue_utils_new
import numpy as np
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--cache_dir")
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--model_name_or_path")
parser.add_argument("--num_train_epochs", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--seed", type=int)
parser.add_argument("--task")
parser.add_argument("--token_vocab")
args = parser.parse_args()

# Load training and evaluation data
data_train = load_dataset(
    'glue', 
    args.task.lower().replace('-', ''), 
    split = 'train', 
    cache_dir = args.cache_dir
)
data_eval = load_dataset(
    'glue', 
    args.task.lower().replace('-', ''), 
    split = 'validation', 
    cache_dir = args.cache_dir
)

# Load GPT2 tokenizer
glue_utils_new.tokenizer = GPT2Tokenizer.from_pretrained(
    args.token_vocab, 
    bos_token = '<|start|>', 
    eos_token = '<|end|>', 
    pad_token = '<pad>',
)

# Add special tokens to tokenizer
single = {'CoLA', 'SST-2'}
NLI = {'QNLI', 'RTE', 'WNLI', 'MNLI'}
similarity = {'MRPC', 'STS-B', 'QQP'}
if args.task in NLI or args.task in similarity:
    glue_utils_new.tokenizer.add_tokens(["<$>"])

# Rename column 'label' to 'labels', which is expected as input of 
# huggingface's models
data_train.rename_column_('label', 'labels')
data_eval.rename_column_('label', 'labels')

# specify columns that can be dropped, as they are not consisting of input of 
# the model
remove_cols = ['idx']
if args.task in {'MRPC', 'STS-B', 'RTE', 'WNLI'}:
    remove_cols += ['sentence1', 'sentence2']
elif args.task == 'QQP':
    remove_cols  += ['question2', 'question2']
elif args.task == 'QNLI':
    remove_cols  += ['question', 'sentence']
elif args.task == 'MNLI':
    remove_cols += ['premise', 'hypothesis']
elif args.task in single:
    remove_cols += ['sentence']

# tokenize the sentences dependent on the task, using the same method that was 
# used in the original GPT
data_train = data_train.map(lambda x: glue_utils_new.encode(x, args.task), 
                            batched = True, remove_columns = remove_cols, 
                            keep_in_memory = True)
data_eval = data_eval.map(lambda x: glue_utils_new.encode(x, args.task), 
                          batched = True, remove_columns = remove_cols, 
                          keep_in_memory = True)

# convert data to torch.tensor
data_train.set_format(type = 'torch')
data_eval.set_format(type = 'torch')

# set seed before creating and shuffling the batches for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# create input batches; the training data is randomly shuffled to implement SGD
batches_train = DataLoader(data_train, sampler = RandomSampler(data_train),
                           batch_size = args.batch_size)
batches_eval = DataLoader(data_eval, sampler = SequentialSampler(data_eval), 
                          batch_size = args.batch_size)

# drop the cached data (only the batches are needed)
data_train.cleanup_cache_files()
data_eval.cleanup_cache_files()

# specify number of classes
n_classes = 1 if task == 'STS-B' else 2

# Instatiate the model:
model = None
## In case of similarity tasks we choose model which processes two sequences
## per input
if task in similarity:
    model = glue_utils_new.GPT2ForSimilarityClassification(
        sequence_size = args.hidden_size*1024,
        n_classes = n_classes,
        gpt_model_name_or_path = args.model_name_or_path,
    )
## For all other tasks we choose model which processes a single sequence
else: 
    model = glue_utils_new.GPT2ForSequenceClassification(
        sequence_size = args.hiddens_size*1024,
        n_classes = n_classes,
        gpt_model_name_or_path = args.model_name_or_path,
    )

# Add new tokens (<start>, <end>) to the embedding matrix
# Weights are randomly initialized, as in GPT paper
model.gpt2model.resize_token_embeddings(len(glue_utils_new.tokenizer))

# Activate CUDA
# model.cuda()

