import argparse
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Tokenizer, GPT2Model

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--eval_data")
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--model_name_or_path")
parser.add_argument("--num_train_epochs", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--seed", type=int)
parser.add_argument("--train_data")
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

# Load training & evaluation data into pandas dataframe.
df_train = pd.read_csv(args.train_data, delimiter='\t', header=0, names=['sentence', 'label'])
df_eval = pd.read_csv(args.eval_data, delimiter='\t', header=0, names=['sentence', 'label'])

# Store sentences and labels as lists.
sentences_train, sentences_eval = df_train.sentence.to_list(), df_eval.sentence.to_list()
labels_train, labels_eval = torch.tensor(df_train.label.to_list()), torch.tensor(df_eval.label.to_list())

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.token_vocab, additional_special_tokens=['<s>','<pad>','</s>'], pad_token='<pad>')

# Calculate length of the longest sentence
max_len = 0
for sent in sentences_train:
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

# Set seed before shuffling the batches for reproducability
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

# ---------------------------------------------------------------------------------------------------------------
# Here we define the classification head of GPT-2 & initialize the model
# ---------------------------------------------------------------------------------------------------------------

# We implement the (simple) approach used for the original GPT.
# That is, the hidden states are fed into a single linear layer to predict the scores.
class GPT2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        sequence_size: int,
        n_classes:int ,
        gpt_model_name_or_path:str,
    ):
        super(GPT2ForSequenceClassification,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name_or_path
        )
        self.lin = nn.Linear(sequence_size, n_classes)

    def forward(self, ids_in, attention_mask):

        gpt_out = self.gpt2model(ids_in, attention_mask = attention_mask)[0]
        n_sentences = gpt_out.shape[0]
        logits = self.lin(gpt_out.view(n_sentences,-1))

        return logits

# Instatiate model
model = GPT2ForSequenceClassification(
    sequence_size = max_len * args.hidden_size,
    n_classes = 2,
    gpt_model_name_or_path = args.model_name_or_path,
)
# Activate CUDA
model.cuda()

# ---------------------------------------------------------------------------------------------------------------
# We choose the same optimizer & hyperparameters used for the other models fine-tuned on GLUE
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

# Define function to compute accuracy for a given batch
def compute_acc(pred, label):
    pred = pred.cpu()
    label = label.cpu()
    return torch.eq(torch.argmax(pred,dim=1),label).sum().item() / len(pred)

# Define loss function 
loss_func = nn.CrossEntropyLoss()

# Initialize list to store training & evaluation history
train_eval_hist = []

# Set seed before training for reproducability
torch.backends.cudnn.deterministic=True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Start training & evaluation loop
for epoch_i in range(0, args.num_train_epochs):
    # Training
    train_loss = 0.0
    model.train()
    for step, batch in enumerate(batches_train):
        input_ids_t = batch[0].to(device)
        attention_mask_t = batch[1].to(device)
        target_t = batch[2].to(device)
        model.zero_grad()
        pred_t = model(input_ids_t, attention_mask_t)
        loss = loss_func(pred_t, target_t)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    batch_train_loss = train_loss / len(batches_train)
    # Evaluation
    model.eval()
    eval_acc = 0
    eval_loss = 0
    for batch in batches_eval:
        input_ids_t = batch[0].to(device)
        attention_mask_t = batch[1].to(device)
        target_t = batch[2].to(device)
        with torch.no_grad():
            pred_t = model(input_ids_t, attention_mask_t)
            loss = loss_func(pred_t, target_t)
        eval_loss += loss.item()
        pred_t = pred_t.detach().cpu()
        target_t = target_t.to('cpu')
        eval_acc += compute_acc(pred_t, target_t)
    batch_eval_acc = eval_acc / len(batches_eval)
    batch_eval_loss = eval_loss / len(batches_eval)
    # Store results of each epoch
    train_eval_hist.append(
        {'epoch': epoch_i + 1,
         'Training Loss': batch_train_loss,
         'Eval Loss': batch_eval_loss,
         'Eval Acc': batch_eval_acc,})


# Create output directory if non-existent
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Save model
torch.save(model.state_dict(), args.output_dir + 'model')

# Save evaluation set results
eval_loss = train_eval_hist[args.num_train_epochs-1].get('Eval Loss')
eval_acc = train_eval_hist[args.num_train_epochs-1].get('Eval Acc')
with open(args.output_dir + 'eval_results_sst-2.txt', "w") as text_file:
    print("eval_loss = {}".format(eval_loss), file=text_file)
    print("eval_acc = {}".format(eval_acc), file=text_file)
    print("epoch = {}".format(args.num_train_epochs), file=text_file)