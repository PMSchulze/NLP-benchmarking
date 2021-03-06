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
parser.add_argument("--task")
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
utils_gpt2_glue.tokenizer = GPT2Tokenizer.from_pretrained(
    args.token_vocab, 
    bos_token = '<|start|>', 
    eos_token = '<|end|>', 
    pad_token = '<pad>',
)

# Add special tokens to tokenizer
single = {'CoLA', 'SST2'}
NLI = {'QNLI', 'RTE', 'WNLI', 'MNLI'}
similarity = {'MRPC', 'STSB', 'QQP'}
if args.task in NLI or args.task in similarity:
    utils_gpt2_glue.tokenizer.add_tokens(["<$>"])

# Rename column 'label' to 'labels', which is the expected keyword argument 
# of huggingface's models.
data_train.rename_column_('label', 'labels')
data_eval.rename_column_('label', 'labels')

# Specify columns that do not contain of input for the model and can thus be
# dropped.  
remove_cols = ['idx']
if args.task in {'MRPC', 'STSB', 'RTE', 'WNLI'}:
    remove_cols += ['sentence1', 'sentence2']
elif args.task == 'QQP':
    remove_cols  += ['question1', 'question2']
elif args.task == 'QNLI':
    remove_cols  += ['question', 'sentence']
elif args.task == 'MNLI':
    remove_cols += ['premise', 'hypothesis']
elif args.task in single:
    remove_cols += ['sentence']

# Tokenize the sentences dependent on the task, using the same method that was 
# used in the original GPT
data_train = data_train.map(lambda x: utils_gpt2_glue.encode(x, args.task), 
                            batched = True, remove_columns = remove_cols) 
data_eval = data_eval.map(lambda x: utils_gpt2_glue.encode(x, args.task), 
                          batched = True, remove_columns = remove_cols) 

# Convert data to torch.tensor
data_train.set_format(type = 'torch')
data_eval.set_format(type = 'torch')

# Set seed before creating and shuffling the batches for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create input batches; the training data is randomly shuffled to implement SGD
batches_train = DataLoader(data_train, sampler = RandomSampler(data_train),
                           batch_size = args.batch_size)
batches_eval = DataLoader(data_eval, sampler = SequentialSampler(data_eval), 
                          batch_size = args.batch_size)

# Drop the cached data (only the batches are needed)
data_train.cleanup_cache_files()
data_eval.cleanup_cache_files()

# Specify number of classes
n_classes = 1 if args.task == 'STSB' else 2

# Instatiate the model:
model = None
## In case of similarity tasks we choose model which processes two sequences
## per input
if args.task in similarity:
    model = utils_gpt2_glue.GPT2ForSimilarityClassification(
        n_classes = n_classes,
        gpt_model_name_or_path = args.model_name_or_path,
    )
## For all other tasks we choose model which processes a single sequence
else: 
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

# ---------------------------------------------------------------------------------------------------------------
# Train & Eval Loop
# ---------------------------------------------------------------------------------------------------------------

# Initialize empty list to store predictions on evaluation set
train_eval_hist = []
logits, true_labels = [], []

# Set seed before training for reproducibility
torch.backends.cudnn.deterministic=True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Start training & evaluation loop
for epoch in range(0, args.num_train_epochs):
    print(f'\nEpoch {epoch+1}/{args.num_train_epochs}')
    print('Training:')
    # Training
    train_loss = 0.0
    model.train()
    for step, batch in enumerate(tqdm(batches_train)):
        # Convert input tensors to cuda device
        inputs_i = {k: v.to(device) for k, v in batch.items()}
        # Set gradients to zero to avoid
        # accumulation of gradients in backward pass
        model.zero_grad()
        # Calculate loss of forward pass
        loss_i = model(**inputs_i)[0]
        # Increment total training loss
        train_loss += loss_i.item()
        # Compute dloss_i/dx for every parameter x
        loss_i.backward()
        # Clip gradients to avoid 'exploding' gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters using gradient descent
        optimizer.step()
        # Update learning rate
        scheduler.step()
    # Calculate the training loss average over all batches
    total_train_loss = train_loss / len(batches_train)
    print('Evaluation:')
    # Evaluation
    model.eval()
    eval_loss = 0
    for batch in tqdm(batches_eval):
        # Convert input tensors to cuda device
        inputs_i = {k: v.to(device) for k, v in batch.items()}
        # Turn off tracking of history, because for evaluation we do not want
        # to perform parameter updates
        with torch.no_grad():
            # Calculate loss and predictions; the latter are later used to 
            # Calculate various task-dependent metrics 
            loss_i, logits_i = model(**inputs_i)
        # Increment evaluation loss 
        eval_loss += loss_i.item()
        # Move predictions to cpu
        logits_i = logits_i.detach().cpu()
        # Move labels to cpu
        true_labels_i = inputs_i['labels'].to('cpu')
        # Add predictions of current batch to list of all predictions
        logits.append(logits_i)
        # Add labels of current batch to list of all labels
        true_labels.append(true_labels_i)
    total_eval_loss = eval_loss / len(batches_eval)
    # Store results of each epoch
    train_eval_hist.append(
        {'epoch': epoch + 1,
         'Training Loss': total_train_loss,
         'Eval Loss': total_eval_loss})

# Load task-specific metrics from huggingface hub 
metric = load_metric(
    'glue', 
    args.task.lower().replace('-', ''), 
    cache_dir = args.cache_dir
)

# Concatenate batches of logits and true labels of last epoch
batchsize_last_epoch = len(batches_eval)
logits = np.concatenate(logits[-batchsize_last_epoch:], axis = 0)
true_labels = np.concatenate(true_labels[-batchsize_last_epoch:], axis = 0)

# If not regression task, then prediction is argmax of logits
preds = np.argmax(logits, axis = 1) if n_classes>1 else logits.flatten()

# Specify predictions and true labels to calculate the scores
metric.add_batch(
    predictions = preds, 
    references = true_labels
)

# Calculate the scores
final_score = metric.compute()

# Create name of task-specific output directory
output_dir_task = os.path.join(args.output_dir, args.task)

# Create the task-specific output directory if not existing already
if not os.path.exists(output_dir_task):
    os.makedirs(output_dir_task)

# Specify path for text file with evaluation results
filepath_out_eval = os.path.join(
    output_dir_task, 
    'eval_results_' + args.task.lower() + '.txt',
)

# Specify path for log file with train/testloss history
filepath_out_log = os.path.join(
    output_dir_task,
    'log_history.json',
)

# Save evaluation results
with open(filepath_out_eval, 'w') as text_file:
    for k,v in final_score.items():
             print(f'{k} = {v}', file = text_file)

# Save log history
with open(filepath_out_log, 'w') as json_file:
    json.dump(train_eval_hist, json_file)
