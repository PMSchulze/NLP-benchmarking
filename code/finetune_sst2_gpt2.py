import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# First we load & prepare training and evaluation data
# ---------------------------------------------------------------------------------------------------------------

# Load the dataset into a pandas dataframe.
df_train = pd.read_csv("/home/ubuntu/data/glue/SST-2/train.tsv", delimiter='\t', header=0, names=['sentence', 'label'])
df_eval = pd.read_csv("/home/ubuntu/data/glue/SST-2/dev.tsv", delimiter='\t', header=0, names=['sentence', 'label'])

# Store sentences and their labels as lists.
sentences_train, sentences_eval = df_train.sentence.to_list(), df_eval.sentence.to_list()
labels_train, labels_eval = df_train.label.to_list(), df_eval.label.to_list()

# Load GPT2 tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('/home/ubuntu/data/token_vocab/roberta/', additional_special_tokens=['<s>','<pad>','</s>'], pad_token='<pad>')

max_len = 0
# For every sentence...
for sent in sentences_train:
    # Tokenize the text and for each sentence return the sequence of indices.
    ids = tokenizer.encode(sent)
    # Update the maximum sentence length.
    max_len = max(max_len, len(ids))

# Tokenize all of the sentences and map the tokens to thier word IDs.
encoding_train = tokenizer(sentences_train, return_tensors='pt', padding=True, truncation=True, max_length = max_len)
encoding_eval = tokenizer(sentences_eval, return_tensors='pt', padding=True, truncation=True, max_length = max_len)
input_ids_train, input_ids_eval = encoding_train['input_ids'], encoding_eval['input_ids']
attention_mask_train, attention_mask_eval = encoding_train['attention_mask'], encoding_eval['attention_mask']

from torch.utils.data import TensorDataset
# Combine the training inputs into a TensorDataset.
dataset_train, dataset_eval = TensorDataset(input_ids_train, attention_mask_train), TensorDataset(input_ids_eval, attention_mask_eval)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
    dataset_train,  # The training samples.
    sampler = RandomSampler(dataset), # Select batches randomly
    batch_size = batch_size # Trains with this batch size.
)
eval_dataloader = DataLoader(
    dataset_eval,  # The training samples.
    sampler = SequentialSampler(dataset), # Select batches sequentially
    batch_size = batch_size # Trains with this batch size.
)

# ---------------------------------------------------------------------------------------------------------------
# Here we define the classification head of GPT-2 & initialize the model
# ---------------------------------------------------------------------------------------------------------------

from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn

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

num_classes = 2
hidden_size = 128
sequence_size = max_len * hidden_size

model = SimpleGPT2SequenceClassifier(
    sequence_size=sequence_size,
    num_classes=num_classes,
    gpt_model_name='/home/ubuntu/lrz_share/models/gpt2/128_2_2_512_10/',
)

# ---------------------------------------------------------------------------------------------------------------
# The optimizer is the same that as we used for other models on GLUE
# ---------------------------------------------------------------------------------------------------------------

from transformers import AdamW
optimizer = AdamW(
    model.parameters(),
    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
)

from transformers import get_linear_schedule_with_warmup

# As in all other cases, we set the number of training epochs to 3.
epochs = 3

# Total number of training steps is [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = 0, # Default value in run_glue.py
    num_training_steps = total_steps
)
------------------------------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------------------------------

def compute_accuracy(y_pred, y_target):
    y_pred = y_pred.cpu()
    y_target = y_target.cpu()
    return torch.eq(torch.argmax(y_pred,dim=1),y_target).sum().item() / len(y_pred)

def make_train_state():
    d = {
        "train_preds": [],
        "train_indexes": [],
        "train_targets": [],
        "train_accuracies": [],
        "train_f1s": [],
        "train_losses": [],
        "val_preds": [],
        "val_indexes": [],
        "val_targets": [],
        "val_accuracies": [],
        "val_f1s": [],
        "val_losses": [],
        "test_preds": [],
        "test_indexes": [],
        "test_targets": [],
        "test_accuracies": [],
        "test_f1s": [],
        "test_losses": [],
        "batch_preds": [],
        "batch_targets": [],
        "batch_indexes": [],
        "epoch_index": 0,
        # "save_path": ''
    }
    return dict(d)
------------------------------------------------------------------------------------------------------------------------------------



loss_func = nn.CrossEntropyLoss()

running_loss = 0.0
running_acc = 0.0
running_f1 = 0.0

model.train()

optimizer.zero_grad()
y_pred=model(input_ids, attention_mask)
y_tgt=torch.tensor([0,1])
loss = loss_func(y_pred, y_tgt)
loss_t = loss.item()

loss.backward()
optimizer.step()

loss_t = loss.item()
# running_loss += (loss_t - running_loss) / (batch_index + 1)

y_pred = y_pred.detach().cpu()
y_tgt = y_tgt.detach().cpu()

acc_t = compute_accuracy(
    y_pred, y_tgt
)

# running_acc += (acc_t - running_acc) / (batch_index + 1)

