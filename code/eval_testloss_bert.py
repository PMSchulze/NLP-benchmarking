import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_corpus")
parser.add_argument("--model_dir")
parser.add_argument("--token_vocab")
args = parser.parse_args()

import torch
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained(args.token_vocab, additional_special_tokens=['<s>','<pad>','</s>','<unk>','<mask>'], pad_token='<pad>')

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(args.model_dir)

with open(args.test_corpus, encoding="utf-8") as f:
    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

batch_encoding = tokenizer(lines, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=128, padding=True)
dataset = batch_encoding["input_ids"]

# there is a bug in trainer.evaluate() which produces CUDA OOM error (see https://github.com/huggingface/transformers/issues/7232)
# therefore we have to split the eval_dataset in pieces
# we split it into pieces of 128 tensors each
dataset_split = torch.split(dataset,128)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    eval_dataset=dataset,
)

test_loss = 0
# here we evaluate the pieces of eval_dataset and average the loss over all pieces
for i in range(0,len(dataset_split)):
    test_loss += trainer.evaluate(dataset_split[i]).get('eval_loss')

test_loss = test_loss/len(dataset_split)

with open(args.model_dir + 'test_loss.txt', "w") as text_file:
    print("test_loss = {}".format(test_loss), file=text_file)
