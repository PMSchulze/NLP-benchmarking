import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--num_hidden_layers", type=int)
parser.add_argument("--num_attention_heads", type=int)
parser.add_argument("--intermediate_size", type=int)
parser.add_argument("--num_train_epochs", type=int)
parser.add_argument("--warmup_steps", type=int)
parser.add_argument("--corpus_pretrain")
parser.add_argument("--output_dir")
parser.add_argument("--token_vocab")
args = parser.parse_args()

import torch

from transformers import RobertaConfig
config = RobertaConfig(vocab_size=30000, hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, num_attention_heads=args.num_attention_heads, intermediate_size=args.intermediate_size, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(args.token_vocab, additional_special_tokens=['<s>','<pad>','</s>','<mask>'])

from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.corpus_pretrain,
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    learning_rate = 1e-4,
    adam_epsilon = 1e-06,
    weight_decay = 0.01,
    warmup_steps = args.warmup_steps,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

trainer.save_model(args.output_dir)
