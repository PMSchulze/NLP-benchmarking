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
batches_eval_matched = DataLoader(data_eval_matched, sampler = SequentialSampler(data_eval_matched), 
                          batch_size = args.batch_size)
batches_eval_mismatched = DataLoader(data_eval_mismatched, sampler = SequentialSampler(data_eval_mismatched), 
                          batch_size = args.batch_size)

# Drop the cached data (only the batches are needed)
data_train.cleanup_cache_files()
data_eval_matched.cleanup_cache_files()
data_eval_mismatched.cleanup_cache_files()

# Instatiate the model:
model = utils_gpt2_glue.GPT2ForSequenceClassification(
    n_classes = 2,
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
logits_matched, true_labels_matched = [], []
logits_mismatched, true_labels_mismatched = [], []

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
    print('Evaluation (matched):')
    # Evaluation
    model.eval()
    eval_loss_matched = 0
    for batch in tqdm(batches_eval_matched):
        # Convert input tensors to cuda device
        inputs_i = {k: v.to(device) for k, v in batch.items()}
        # Turn off tracking of history, because for evaluation we do not want
        # to perform parameter updates
        with torch.no_grad():
            # Calculate loss and predictions; the latter are later used to 
            # Calculate various task-dependent metrics 
            loss_i, logits_i = model(**inputs_i)
        # Increment evaluation loss 
        eval_loss_matched += loss_i.item()
        # Move predictions to cpu
        logits_i = logits_i.detach().cpu()
        # Move labels to cpu
        true_labels_i = inputs_i['labels'].to('cpu')
        # Add predictions of current batch to list of all predictions
        logits_matched.append(logits_i)
        # Add labels of current batch to list of all labels
        true_labels_matched.append(true_labels_i)
    total_eval_loss_matched = eval_loss_matched / len(batches_eval_matched)
    print('Evaluation (mismatched):')
    # Evaluation
    model.eval()
    eval_loss_mismatched = 0
    for batch in tqdm(batches_eval_mismatched):
        # Convert input tensors to cuda device
        inputs_i = {k: v.to(device) for k, v in batch.items()}
        # Turn off tracking of history, because for evaluation we do not want
        # to perform parameter updates
        with torch.no_grad():
            # Calculate loss and predictions; the latter are later used to 
            # Calculate various task-dependent metrics 
            loss_i, logits_i = model(**inputs_i)
        # Increment evaluation loss 
        eval_loss_mismatched += loss_i.item()
        # Move predictions to cpu
        logits_i = logits_i.detach().cpu()
        # Move labels to cpu
        true_labels_i = inputs_i['labels'].to('cpu')
        # Add predictions of current batch to list of all predictions
        logits_mismatched.append(logits_i)
        # Add labels of current batch to list of all labels
        true_labels_mismatched.append(true_labels_i)
    total_eval_loss_mismatched = eval_loss_mismatched / len(batches_eval_matched)
    # Store results of each epoch
    train_eval_hist.append(
        {'epoch': epoch + 1,
         'Training Loss': total_train_loss,
         'Eval Loss (matched)': total_eval_loss_matched,
         'Eval Loss (mismatched)': total_eval_loss_mismatched})

# Load task-specific metrics from huggingface hub 
metric_matched = load_metric(
    'glue', 
    args.task.lower().replace('-', ''), 
    cache_dir = args.cache_dir
)
metric_mismatched = load_metric(
    'glue', 
    args.task.lower().replace('-', ''), 
    cache_dir = args.cache_dir
)

# Concatenate batches of logits and true labels of last epoch
batchsize_last_epoch_matched = len(batches_eval_matched)
batchsize_last_epoch_mismatched = len(batches_eval_mismatched)
logits_matched = np.concatenate(
    logits_matched, 
    axis = 0
)[-batchsize_last_epoch_matched:]
true_labels_matched = np.concatenate(
    true_labels_matched, 
    axis = 0
)[-batchsize_last_epoch_matched:]
logits_mismatched = np.concatenate(
    logits_mismatched, 
    axis = 0
)[-batchsize_last_epoch_mismatched:]
true_labels_mismatched = np.concatenate(
    true_labels_mismatched, 
    axis = 0
)[-batchsize_last_epoch_mismatched:]

# If not regression task, then prediction is argmax of logits
preds_matched = np.argmax(logits_matched, axis = 1) if n_classes>1 \
    else logits_matched.flatten()
preds_mismatched = np.argmax(logits_mismatched, axis = 1) if n_classes>1 \
    else logits_mismatched.flatten()

# Specify predictions and true labels to calculate the scores
metric_matched.add_batch(
    predictions = preds_matched, 
    references = true_labels_matched
)
metric_mismatched.add_batch(
    predictions = preds_mismatched, 
    references = true_labels_mismatched
)

# Calculate the scores
final_score_matched = metric_matched.compute()
final_score_mismatched = metric_mismatched.compute()

# Create name of task-specific output directory
output_dir_task = os.path.join(args.output_dir, args.task)

# Create the task-specific output directory if not existing already
if not os.path.exists(output_dir_task):
    os.makedirs(output_dir_task)

# Specify path for text file with evaluation results
filepath_out_eval_matched = os.path.join(
    output_dir_task, 
    'eval_results_' + args.task.lower() + '_matched.txt',
)
# Specify path for text file with evaluation results
filepath_out_eval_mismatched = os.path.join(
    output_dir_task, 
    'eval_results_' + args.task.lower() + '_mismatched.txt',
)

# Specify path for log file with train/testloss history
filepath_out_log = os.path.join(
    output_dir_task,
    'log_history.json',
)

# Save evaluation results
with open(filepath_out_eval_matched, 'w') as text_file:
    for k,v in final_score.items():
             print(f'{k} = {v}', file = text_file)

with open(filepath_out_eval_mismatched, 'w') as text_file:
    for k,v in final_score.items():
             print(f'{k} = {v}', file = text_file)

# Save log history
with open(filepath_out_log, 'w') as json_file:
    json.dump(train_eval_hist, json_file)
