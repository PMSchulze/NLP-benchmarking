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

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained(args.token_vocab, additional_special_tokens=['<pad>'], pad_token='<pad>')

from transformers import GPT2Config
config = GPT2Config(vocab_size=len(tokenizer), n_embd = args.hidden_size, n_layer = args.num_hidden_layers, n_head = args.num_attention_heads, 
                    bos_token_id=29999, eos_token_id=29999, n_inner = args.intermediate_size, resid_pdrop=0.1, embd_pdrop=0.1, 
                    attn_pdrop=0.1, activation_function = 'gelu')

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel(config=config)

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.corpus_pretrain,
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    learning_rate = 2.5e-4,
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

