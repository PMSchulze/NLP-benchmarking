import torch
import torch.nn as nn
from transformers.modeling_utils import Conv1D, PreTrainedModel
from transformers.modeling_gpt2 import (
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers import GPT2Tokenizer

# Load GPT2 tokenizer
tokenizer = None

def encode_single(examples):
    modified_input = ['<|start|>'+ x + '<|end|>' for x in examples['sentence']]
    return tokenizer(
        modified_input, 
        truncation = True, 
        padding = 'max_length',
        max_length = 512
    )

def encode_nli(examples, task):
    part1, part2 = '', ''
    if task == 'QNLI':
        part1, part2 = 'question', 'sentence'
    elif task == 'MNLI':
        part1, part2 = 'premise', 'hypothesis'
    else:
        part1, part2 = 'sentence1', 'sentence2'
    modified_input = [
        '<|start|>'+ x + '<$>' + y + '<|end|>' 
        for x,y in zip(examples[part1], examples[part2])
    ]
    tok = tokenizer(
        modified_input, 
        truncation = True, 
        padding = 'max_length',
        max_length = 512
    )
    return tok

def encode_similarity(examples, task):
    part1, part2 = (
        'sentence1' if task != 'QQP' else 'question1', 
        'sentence2' if task != 'QQP' else 'question2'
    )
    modified_input1 = [
        '<|start|>'+ x + '<$>' + y + '<|end|>' 
        for x,y in zip(examples[part1], examples[part2])
    ]
    modified_input2 = [
        '<|start|>'+ y + '<$>' + x + '<|end|>' 
        for x,y in zip(examples[part1], examples[part2])
    ]
    tok1 = tokenizer(
        modified_input1, 
        truncation = True, 
        padding ='max_length', 
        max_length = 512
    )
    tok2 = tokenizer(
        modified_input2, 
        truncation = True, 
        padding = 'max_length', 
        max_length = 512
    )
    out = {
        'attention_mask1': tok1['attention_mask'],
        'attention_mask2': tok2['attention_mask'],
        'input_ids1': tok1['input_ids'],
        'input_ids2': tok2['input_ids'],
    }
    return out
    
def encode(examples, task):
    single = {'CoLA', 'SST2'}
    nli = {'QNLI', 'RTE', 'WNLI', 'MNLI'}
    similarity = {'MRPC', 'STSB', 'QQP'}
    if task in single:
        return encode_single(examples)
    elif task in similarity:
        return encode_similarity(examples, task)
    else:
        return encode_nli(examples, task)

# -----------------------------------------------------------------------------
# Here we define the classification head of GPT-2 & initialize the model
# -----------------------------------------------------------------------------

# We implement the approach used for the original GPT.
#
# That is, the hidden states which are obtained by the transformer (GPT2Model) 
# are fed into a single linear layer to predict the scores.
#
# For similarity tasks the authors of GPT feed two sequences into separate
# transformers for each input. The resulting hidden states are then added
# before fed into the linear layer.


# This head takes a single sequence as input.
# It is used for all GlUE tasks except for similarity/paraphrasing. 
class GPT2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        n_classes:int ,
        gpt_model_name_or_path:str,
    ):
        super(GPT2ForSequenceClassification,self).__init__()
        
        # Load the pre-trained transformer
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name_or_path
        )
        # Load the hidden size from the config
        hidden_size = self.gpt2model.config.hidden_size
        # Store number of classes to predict
        self.n_classes = n_classes
        # Define dense linear layer
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Define dropout with p=0.1
        self.dropout = nn.Dropout(0.1)
        # Define a linear layer which predicts scores from hidden states
        self.out_proj = nn.Linear(hidden_size, n_classes) 
        # Initialize weights 
        # self.apply(self.init_weights)

    # Define function to initialize weights as in other huggingface models
    #def init_weights(self, m):
    #    if isinstance(m, (nn.Linear, nn.Embedding, Conv1D)):
    #        m.weight.data.normal_(mean=0.0, std=0.02)
    #        if isinstance(m, (nn.Linear, Conv1D)) and m.bias is not None:
    #            m.bias.data.zero_()
    #    elif isinstance(m, nn.LayerNorm):
    #        m.bias.data.zero_()
    #        m.weight.data.fill_(1.0)

    def forward(self, attention_mask, input_ids, labels):
        
        # Compute the hidden states of all tokens for pre-trained model
        gpt_out_all = self.gpt2model(
            input_ids, 
            attention_mask = attention_mask
        )[0]
        # Obtain the positions of the last tokens before pad token (which is 1)
        #sequence_lengths = torch.ne(input_ids, 1).sum(-1) - 1
        # Extract the hidden states of the last token for each sequence
        #x = gpt_out_all[torch.arange(gpt_out_all.size(0)), sequence_lengths]
        # Apply dropout
        #x = self.dropout(x)
        # Apply dense layer
        #x = self.dense(x)
        # Apply tanh activation
        #x = torch.tanh(x)
        # Apply dropout
        #x = self.dropout(x)
        # Compute logits
        #logits = self.out_proj(x)
        
        logits = self.out_proj(gpt_out_all)
        sequence_lengths = torch.ne(input_ids, 1).sum(-1) - 1
        logits = logits[range(gpt_out_all.size(0)), sequence_lengths]
        
        loss = None
        # Use MSE loss for regression tasks (STSB) 
        if self.n_classes == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        # Use cross entropy for classification tasks (all but STSB)
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
        n_classes:int ,
        gpt_model_name_or_path:str,
    ):

        super(GPT2ForSimilarityClassification,self).__init__()
        # Load the pre-trained transformer
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name_or_path
        )
        # Load the hidden size from the config
        hidden_size = self.gpt2model.config.hidden_size
        # Store number of classes to predict
        self.n_classes = n_classes
        # Define dense linear layer
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Define dropout with p=0.1
        self.dropout = nn.Dropout(0.1)
        # Define a linear layer which predicts scores from hidden states
        self.out_proj = nn.Linear(hidden_size, n_classes) 
        # Initialize weights 
        self.apply(self.init_weights)

    # Define function to initialize weights as in other huggingface models
    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding, Conv1D)):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, (nn.Linear, Conv1D)) and m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
    
    def forward(
        self, 
        attention_mask1, 
        attention_mask2, 
        input_ids1, 
        input_ids2, 
        labels
    ):
        
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
        # Obtain the positions of the last tokens before pad token (which is 1)
        sequence_lengths = torch.ne(input_ids1, 1).sum(-1) - 1
        # Extract hidden states of the last tokens for first sequence
        x1 = gpt_out_all1[torch.arange(gpt_out_all1.size(0)), sequence_lengths]
        # Extract hidden states of the last tokens for second sequence
        x2 = gpt_out_all2[torch.arange(gpt_out_all2.size(0)), sequence_lengths]
        # Add the two hidden states element-wise
        x = x1 + x2
        # Apply dropout
        x = self.dropout(x)
        # Apply dense layer
        x = self.dense(x)
        # Apply tanh activation
        x = torch.tanh(x)
        # Apply dropout
        x = self.dropout(x)
        # Compute logits
        logits = self.out_proj(x)
        
        loss = None
        # Use MSE loss for regression tasks (STSB) 
        if self.n_classes == 1:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        # Use cross entropy for classification tasks (all but STSB)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
        
        # Return loss and logits for each forward pass
        return loss, logits
