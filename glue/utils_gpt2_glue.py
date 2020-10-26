import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

# Load GPT2 tokenizer
tokenizer = None

def encode_single(examples):
    modified_input = ['<|start|>'+ x + '<|end|>' for x in examples['sentence']]
    return tokenizer(modified_input, truncation = True, padding = 'max_length', max_length = 1024)

def encode_nli(examples, task):
    part1, part2 = '', ''
    if task == 'QNLI':
        part1, part2 = 'question', 'sentence'
    elif task == 'MNLI':
        part1, part2 = 'premise', 'hypothesis'
    else:
        part1, part2 = 'sentence1', 'sentence2'
    modified_input = ['<|start|>'+ x + '<$>' + y + '<|end|>' for x,y in zip(examples[part1], examples[part2])]
    tok = tokenizer(modified_input, truncation = True, padding = 'max_length', max_length = 1024)
    return tok

def encode_similarity(examples, task):
    part1, part2 = 'sentence1' if task != 'QQP' else 'question1', 'sentence2' if task != 'QQP' else 'question2'
    modified_input1 = ['<|start|>'+ x + '<$>' + y + '<|end|>' for x,y in zip(examples[part1], examples[part2])]
    modified_input2 = ['<|start|>'+ y + '<$>' + x + '<|end|>' for x,y in zip(examples[part1], examples[part2])]
    tok1 = tokenizer(modified_input1, truncation = True, padding = 'max_length', max_length = 1024)
    tok2 = tokenizer(modified_input2, truncation = True, padding = 'max_length', max_length = 1024)
    out = {
        'attention_mask1': tok1['attention_mask'],
        'attention_mask2': tok2['attention_mask'],
        'input_ids1': tok1['input_ids'],
        'input_ids2': tok2['input_ids'],
    }
    return out
    
def encode(examples, task):
    single = {'CoLA', 'SST-2'}
    nli = {'QNLI', 'RTE', 'WNLI', 'MNLI'}
    similarity = {'MRPC', 'STS-B', 'QQP'}
    if task in single:
        return encode_single(examples)
    elif task in similarity:
        return encode_similarity(examples, task)
    else:
        return encode_nli(examples, task)

# ---------------------------------------------------------------------------------------------------------------
# Here we define the classification head of GPT-2 & initialize the model
# ---------------------------------------------------------------------------------------------------------------

# We implement the approach used for the original GPT.
#
# That is, the hidden states which are obtained by the transformer (GPT2Model) 
# are fed into a single linear layer to predict the scores.
#
# For similarity tasks the authors of GPT feed two sequences into separate
# transformers for each input. The resulting hidden states are then added
# before fed into the linear layer.


class GPT2Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.lin(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# This head takes a single sequence as input.
# It is used for all GlUE tasks except for similarity/paraphrasing. 
class GPT2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_classes:int ,
        gpt_model_name_or_path:str,
    ):
        super(GPT2ForSequenceClassification,self).__init__()
        
        # Load the pre-trained transformer
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name_or_path
        )
        # Define a linear layer which predicts scores from hidden states
        self.lin = nn.Linear(hidden_size, n_classes)
        self.n_classes = n_classes

    def forward(self, attention_mask, input_ids, labels):
        
        # Compute the hidden states of all tokens for pre-trained model
        gpt_out_all = self.gpt2model(
            input_ids, 
            attention_mask = attention_mask
        )[0]
        # Extract the hidden states of the first token
        gpt_out_first = gpt_out_all[:,0,:]
        # Calculate logits for batch 
        logits = self.lin(gpt_out_first)
        
        loss = None
        # Use MSE loss for regression tasks (SST-2) 
        if self.n_classes == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        # Use cross entropy for classification tasks (all but SST-2)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))

        # Return loss and logits for each forward pass
        return loss, logits


# This head takes two sequences as input, each processed by an individual
# transformers. It is used for all similarity/paraphrasing tasks.
class GPT2ForSimilarityClassification(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_classes:int ,
        gpt_model_name_or_path:str,
    ):

        # Load the pre-trained transformer
        super(GPT2ForSimilarityClassification,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name_or_path
        )
        # Define a linear layer which predicts scores from hidden states
        self.lin = nn.Linear(hidden_size, n_classes)
        self.n_classes = n_classes

    def forward(self, attention_mask1, attention_mask2, input_ids1, input_ids2, labels):
        
        # Calculate hidden states of all tokens for the first sequence
        gpt_out_all1 = self.gpt2model(
            input_ids1, 
            attention_mask = attention_mask1
        )[0]
        # Calculate hidden states of all tokens for the second sequence
        gpt_out_all2 = self.gpt2model(
            input_ids2, 
            attention_mask = attention_mask2
        )[0]
        # Extract hidden states of the first token for first sequence
        gpt_out_first1 = gpt_out_all1[:,0,:]
        # Extract hidden states of the first token for second sequence
        gpt_out_first2 = gpt_out_all2[:,0,:]
        # Add the two hidden states element-wise
        gpt_out_first = gpt_out_first1 + gpt_out_first2
        # Calculate logits for batch
        logits = self.lin(gpt_out_first)

        loss = None
        # Use MSE loss for regression tasks (SST-2) 
        if self.n_classes == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        # Use cross entropy for classification tasks (all but SST-2)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
        
        # Return loss and logits for each forward pass
        return loss, logits
