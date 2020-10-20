import argparse from glue_utils import load_data, extract_and_prepare
import json
import numpy as np
import os
import os.path
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Tokenizer, GPT2Model

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--glue_dir")
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--model_name_or_path")
parser.add_argument("--num_train_epochs", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--seed", type=int)
parser.add_argument("--task")
parser.add_argument("--token_vocab")
args = parser.parse_args()

# Setup CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------------------------------------------------------------
# First we load & prepare training and evaluation data
# ---------------------------------------------------------------------------------------------------------------

# Load training & evaluation data into pandas dataframe
df_train, df_eval = load_data(args.task, args.glue_dir)

# Extract labels and sentences, which are modified by adding task-specific special tokens
labels_train, sentences_train = extract_and_prepare(args.task, df_train)
labels_eval, sentences_eval = extract_and_prepare(args.task, df_eval)

# Convert labels and store as them torch tensor
le = preprocessing.LabelEncoder()
labels_train = torch.tensor(le.fit_transform(labels_train))
labels_eval = torch.tensor(le.fit_transform(labels_eval))

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.token_vocab, bos_token = '<|start|>', eos_token = '<|end|>', pad_token='<pad>')

# Add special tokens to tokenizer
NLI = {'QNLI', 'RTE', 'WNLI', 'MNLI'}
similarity = {'MRPC', 'STS-B', 'QQP'}
if args.task in NLI:
    tokenizer.add_tokens(["<$>"])
    
# Calculate length of the longest sentence
max_len = 0
for sent in sentences_train+sentences_eval:
    ids = tokenizer.encode(sent)
    max_len = max(max_len, len(ids))

# Tokenize all sentences & store IDs and attention mask
encoding_train = tokenizer(sentences_train, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len)
encoding_eval = tokenizer(sentences_eval, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len)
input_ids_train, input_ids_eval = encoding_train['input_ids'], encoding_eval['input_ids']
attention_mask_train, attention_mask_eval = encoding_train['attention_mask'], encoding_eval['attention_mask']

# Store IDs, attention masks and labels in one object
data_train = TensorDataset(input_ids_train, attention_mask_train, labels_train)
data_eval = TensorDataset(input_ids_eval, attention_mask_eval, labels_eval)

# Set seed before shuffling the batches for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create input batches
batches_train = DataLoader(
    data_train, 
    sampler = RandomSampler(data_train),
    batch_size = args.batch_size
)
batches_eval = DataLoader(
    data_eval,
    sampler = SequentialSampler(data_eval),
    batch_size = args.batch_size
)

# Instatiate model 
model = GPT2ForSequenceClassification(
    sequence_size = max_len * args.hidden_size,
    n_classes = len(torch.unique(labels_train)),
    gpt_model_name_or_path = args.model_name_or_path,
)

# Add new tokens (<start>, <end>) to the embedding matrix
# Weights are randomly initialized, as in GPT paper
model.gpt2model.resize_token_embeddings(len(tokenizer)) 

# Activate CUDA
model.cuda()

# ---------------------------------------------------------------------------------------------------------------
# We choose the same optimizer & hyperparameters that we used for BERT and RoBERTa on GLUE
# ---------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------
# Train & Eval Loop
# ---------------------------------------------------------------------------------------------------------------

# Define loss function 
loss_func = nn.CrossEntropyLoss()

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
for epoch_e in range(0, args.num_train_epochs):
    # Training
    train_loss = 0.0
    model.train()
    for step, batch in enumerate(batches_train):
        input_ids_i = batch[0].to(device)
        attention_mask_i = batch[1].to(device)
        true_labels_i = batch[2].to(device)
        model.zero_grad()
        preds_i = model(input_ids_i, attention_mask_i)
        loss = loss_func(preds_i, true_labels_i)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    batch_train_loss = train_loss / len(batches_train)
    # Evaluation
    model.eval()
    eval_loss = 0
    for batch in batches_eval:
        input_ids_i = batch[0].to(device)
        attention_mask_i = batch[1].to(device)
        true_labels_i = batch[2].to(device)
        with torch.no_grad():
            preds_i = model(input_ids_i, attention_mask_i)
            loss = loss_func(preds_i, true_labels_i)
        eval_loss += loss.item()
        preds_i = preds_i.detach().cpu()
        true_labels_i = true_labels_i.to('cpu')
        logits.append(preds_i)
        true_labels.append(true_labels_i)
    batch_eval_loss = eval_loss / len(batches_eval)
    # Store results of each epoch
    train_eval_hist.append(
        {'epoch': epoch_e + 1,
         'Training Loss': batch_train_loss,
         'Eval Loss': batch_eval_loss})


# Convert logits to predictions and append over all batches
predictions = np.argmax(np.concatenate(logits, axis=0),axis=1)
true_labels = np.concatenate(true_labels, axis=0)

# Store evaluation loss
eval_loss = train_eval_hist[args.num_train_epochs-1].get('Eval Loss')

# ---------------------------------------------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------------------------------------------

# Set name of output_dir dependent on task 
output_dir = os.path.join(args.output_dir, args.task)

# Create output directory if not existing
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save model
torch.save(model.state_dict(), os.path.join(output_dir, 'model'))

# Define function to compute accuracy
def compute_acc(preds, labels):
    return (preds == labels).mean()

# Save evaluation set results
if args.task in NLI or args.task == 'SST-2':
    eval_acc = compute_acc(predictions, true_labels)
    with open(os.path.join(output_dir, 'eval_results_' + args.task.lower() + '.txt'), "w") as text_file:
        print("eval_loss = {}".format(eval_loss), file=text_file)
        print("eval_acc = {}".format(eval_acc), file=text_file)
        print("epoch = {}".format(args.num_train_epochs), file=text_file)
elif args.task == 'CoLA':
    eval_mcc = matthews_corrcoef(predictions,true_labels)
    with open(os.path.join(output_dir, 'eval_results_cola.txt'), "w") as text_file:
        print("eval_loss = {}".format(eval_loss), file=text_file)
        print("eval_mcc = {}".format(eval_mcc), file=text_file)
        print("epoch = {}".format(args.num_train_epochs), file=text_file)
        
# Save training history
with open(os.path.join(output_dir, 'train_eval_hist.json'), 'w') as json_file:
    json.dump(train_eval_hist, json_file)