import torch

from transformers import GPT2Config
config = GPT2Config(vocab_size=30_000, n_embd = 384, n_layer = 6, n_head = 6, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel(config=config)

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained('/home/ubuntu/data/token_vocab/gpt2/', additional_special_tokens=['<s>','<pad>','</s>','<unk>','<mask>'], pad_token='<pad>')

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='/home/ubuntu/data/pretrain_data/wiki_train.txt',
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="/home/ubuntu/models/gpt2/",
    overwrite_output_dir=True,
    learning_rate = 2.5e-4,
    adam_epsilon = 1e-06,
    weight_decay = 0.01,
    warmup_steps = 2000,
    num_train_epochs=1,
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

trainer.save_model("/home/ubuntu/models/gpt2/")
