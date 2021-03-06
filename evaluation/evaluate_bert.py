import argparse
import os
import torch
from transformers import (
     BertTokenizerFast,
     DataCollatorForNextSentencePrediction,
     TextDatasetForNextSentencePrediction,
     BertForPreTraining,
     Trainer
)

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_eval")
parser.add_argument("--block_size", type = int)
parser.add_argument("--model_name_or_path")
parser.add_argument("--token_vocab", default = '/home/ubuntu/lrz_share/data/token_vocab/bert/')
args = parser.parse_args()

tokenizer = BertTokenizerFast.from_pretrained(args.token_vocab)

model = BertForPreTraining.from_pretrained(args.model_name_or_path)

data_eval = TextDatasetForNextSentencePrediction(
    tokenizer = tokenizer,
    file_path = args.corpus_eval, 
    block_size = args.block_size,
)

data_collator = DataCollatorForNextSentencePrediction(
    tokenizer = tokenizer, 
    mlm = True, 
    mlm_probability = 0.15,
    block_size = args.block_size
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
