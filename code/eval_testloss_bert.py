import torch
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('/home/ubuntu/data/token_vocab/bert/')

from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('/home/ubuntu/models/bert/128_10_2_512_10/')


with open('/home/ubuntu/data/pretrain_data/wiki_test.txt', encoding="utf-8") as f:
    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

batch_encoding = tokenizer(lines, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=128, padding=True)
dataset = batch_encoding["input_ids"]

# there is a bug in trainer.evaluate() which produces CUDA OOM error (see https://github.com/huggingface/transformers/issues/7232)
# therefore we have to split the eval_dataset in pieces 
dataset_split = torch.split(dataset,200)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
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
