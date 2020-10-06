import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Tokenizer, GPT2Model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval_data")
parser.add_argument("--train_data")
parser.add_argument("--token_vocab")
parser.add_argument("--batch_size")
parser.add_argument("--seed")
parser.add_argument("--token_vocab")
parser.add_argument("--hidden_size")
parser.add_argument("--model_name_or_path")
parser.add_argument("--num_train_epochs")
parser.add_argument("--output_dir")
args = parser.parse_args()

# Setup CUDA
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------------------------------------------------------------
# First we load & prepare training and evaluation data
# ---------------------------------------------------------------------------------------------------------------

# Load the dataset into a pandas dataframe.
df_train = pd.read_csv(args.train_data, delimiter='\t', header=0, names=['sentence', 'label'])
df_eval = pd.read_csv(args.eval_data, delimiter='\t', header=0, names=['sentence', 'label'])

# Store sentences and their labels as lists.
sentences_train, sentences_eval = df_train.sentence.to_list(), df_eval.sentence.to_list()
labels_train, labels_eval = torch.tensor(df_train.label.to_list()), torch.tensor(df_eval.label.to_list())

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.token_vocab, additional_special_tokens=['<s>','<pad>','</s>'], pad_token='<pad>')

max_len = 0
# For every sentence...
for sent in sentences_train:
    # Tokenize the text and for each sentence return the sequence of indices.
    ids = tokenizer.encode(sent)
    # Update the maximum sentence length.
    max_len = max(max_len, len(ids))

# Tokenize all of the sentences and map the tokens to thier word IDs.
encoding_train = tokenizer(sentences_train, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len)
encoding_eval = tokenizer(sentences_eval, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len)
input_ids_train, input_ids_eval = encoding_train['input_ids'], encoding_eval['input_ids']
attention_mask_train, attention_mask_eval = encoding_train['attention_mask'], encoding_eval['attention_mask']

# Combine the training inputs into a TensorDataset.
dataset_train = TensorDataset(input_ids_train, attention_mask_train, labels_train)
dataset_eval = TensorDataset(input_ids_eval, attention_mask_eval, labels_eval)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

batch_size = args.batch_size

set_seed(args.seed)
# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
    dataset_train,  # The training samples.
    sampler = RandomSampler(dataset_train), # Select batches randomly
    batch_size = batch_size # Trains with this batch size.
)
eval_dataloader = DataLoader(
    dataset_eval,  # The training samples.
    sampler = SequentialSampler(dataset_eval), # Select batches sequentially
    batch_size = batch_size # Trains with this batch size.
)

# ---------------------------------------------------------------------------------------------------------------
# Here we define the classification head of GPT-2 & initialize the model
# ---------------------------------------------------------------------------------------------------------------

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(
        self, 
        sequence_size: int,
        num_classes:int ,
        gpt_model_name:str, 
    ):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name
        )
        self.fc1 = nn.Linear(sequence_size, num_classes)
        
    def forward(self, x_in, attention_mask):
        
        gpt_out = self.gpt2model(x_in, attention_mask=attention_mask)[0]
        batch_size = gpt_out.shape[0]
        prediction_vector = self.fc1(gpt_out.view(batch_size,-1))
    
        return prediction_vector

model = SimpleGPT2SequenceClassifier(
    sequence_size=max_len * args.hidden_size,
    num_classes=2,
    gpt_model_name=args.model_name_or_path,
)
model.cuda()

# ---------------------------------------------------------------------------------------------------------------
# We choose the same optimizer (and hyperparam.) that was used for the other models fine-tuned on GLUE
# ---------------------------------------------------------------------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr = 2e-5, 
    eps = 1e-8
)

# Total number of training steps is [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * args.num_train_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = 0, # Default value in run_glue.py
    num_training_steps = total_steps
) 

# ---------------------------------------------------------------------------------------------------------------
# Train & Eval Loop
# ---------------------------------------------------------------------------------------------------------------

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

def compute_accuracy(y_pred, y_target):
    y_pred = y_pred.cpu()
    y_target = y_target.cpu()
    return torch.eq(torch.argmax(y_pred,dim=1),y_target).sum().item() / len(y_pred)

loss_func = nn.CrossEntropyLoss()
training_stats = []

torch.backends.cudnn.deterministic=True
set_seed(42)

for epoch_i in range(0, args.num_train_epochs):
    # Training
    training_loss = 0.0
    model.train()    
    for step, batch in enumerate(train_dataloader):
        input_ids_t = batch[0].to(device)
        attention_mask_t = batch[1].to(device)
        target_t = batch[2].to(device)
        model.zero_grad()
        pred_t = model(input_ids_t, attention_mask_t)
        loss = loss_func(pred_t, target_t)
        training_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = training_loss / len(train_dataloader)
    # Validation
    model.eval()
    eval_accuracy = 0
    eval_loss = 0
    for batch in eval_dataloader:
        input_ids_t = batch[0].to(device)
        attention_mask_t = batch[1].to(device)
        target_t = batch[2].to(device)
        with torch.no_grad():        
            pred_t = model(input_ids_t, attention_mask_t)
            loss = loss_func(pred_t, target_t)
        eval_loss += loss.item()
        pred_t = pred_t.detach().cpu()
        target_t = target_t.to('cpu')
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        eval_accuracy += compute_accuracy(pred_t, target_t)
    avg_eval_accuracy = eval_accuracy / len(eval_dataloader)
    avg_eval_loss = eval_loss / len(eval_dataloader)    
    # Record all statistics from this epoch.
    training_stats.append(
        {'epoch': epoch_i + 1,
         'Training Loss': avg_train_loss,
         'Valid. Loss': avg_eval_loss,
         'Valid. Accur.': avg_eval_accuracy,})

 
# Save model
torch.save(model.state_dict(), args.output_dir + 'model')

# Save evaluation set results
eval_loss=training_stats[args.num_train_epochs-1].get('Valid. Loss')
eval_acc=training_stats[args.num_train_epochs-1].get('Valid. Accur.')
with open(args.output_dir + 'eval_results_sst-2.txt', "w") as text_file:
    print("eval_loss = {}".format(eval_loss), file=text_file)
    print("eval_acc = {}".format(eval_acc), file=text_file)
    print("epoch = {}".format(args.num_train_epochs), file=text_file)


