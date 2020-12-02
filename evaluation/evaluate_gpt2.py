import argparse
import os
import torch
from pretrain_utils import LineByLineTextDatasetCached
from transformers import (
     GPT2TokenizerFast,
     DataCollatorForLanguageModeling,
     GPT2LMHeadModel,
     Trainer
)

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_eval")
parser.add_argument("--block_size", type = int)
parser.add_argument("--model_name_or_path")
parser.add_argument("--token_vocab", default = '/home/ubuntu/lrz_share/data/token_vocab/gpt2/')
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained(
    args.token_vocab,
    additional_special_tokens=['<pad>'],
    pad_token='<pad>'
)

model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

from pretrain_utils import LineByLineTextDatasetCached
data_eval = LineByLineTextDatasetCached(
    tokenizer = tokenizer,
    file_path = args.corpus_eval, 
    block_size = args.block_size,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, 
    mlm = False, 
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    eval_dataset=data_eval,
    prediction_loss_only=True
)

eval_loss = trainer.evaluate().get('eval_loss')

with open(os.path.join(args.model_name_or_path, 'eval_loss_final.txt'), 'w') as f:
    print("eval_loss: {}".format(eval_loss), file = f)
