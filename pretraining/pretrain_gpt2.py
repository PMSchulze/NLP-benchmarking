import argparse
import os.path
import pickle
from pretrain_utils import write_time
import time
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments
)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type = int)
parser.add_argument("--num_hidden_layers", type = int)
parser.add_argument("--num_attention_heads", type = int)
parser.add_argument("--intermediate_size", type = int)
parser.add_argument("--num_train_epochs", type = int)
parser.add_argument("--block_size", type = int)
parser.add_argument("--batch_size", type = int)
parser.add_argument("--warmup_steps", type = int)
parser.add_argument("--corpus_train")
parser.add_argument("--corpus_eval")
parser.add_argument("--output_dir")
parser.add_argument("--token_vocab")
parser.add_argument("--seed", type = int)

parser.add_argument(
    "--attention_probs_dropout_prob", type = float, default = 0.1
)
parser.add_argument("--resid_pdrop", type = float, default = 0.1) 
parser.add_argument("--embd_pdrop", type = float, default = 0.1) 
parser.add_argument("--learning_rate", type = float, default = 2.5e-4) 
parser.add_argument("--adam_epsilon", type = float, default = 1e-06) 
parser.add_argument("--adam_beta1", type = float, default = 0.9) 
parser.add_argument("--adam_beta2", type = float, default = 0.999) 
parser.add_argument("--weight_decay", type = float, default = 0.01) 
parser.add_argument("--long_range", type = bool, default = False) 

args = parser.parse_args()


tokenizer = GPT2TokenizerFast.from_pretrained(
    args.token_vocab,
    additional_special_tokens=['<pad>'],
    pad_token='<pad>'
)

data_train = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = args.corpus_train, 
    block_size = args.block_size,
)

data_eval = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = args.corpus_eval, 
    block_size = args.block_size,
)

config = GPT2Config(
    vocab_size = len(tokenizer),
    n_embd = args.hidden_size,
    n_layer = args.num_hidden_layers,
    n_head = args.num_attention_heads,
    n_inner = args.intermediate_size,
    attn_pdrop = args.attention_probs_dropout_prob,
    resid_pdrop=args.resid_pdrop,
    embd_pdrop=args.embd_pdrop,
    activation_function = 'gelu',
    bos_token_id = 29999, 
    eos_token_id = 29999,
    n_positions = 512,
    n_ctx = 512,
)

if args.long_range:
    model= GPT2LMHeadModel.from_pretrained(
        os.path.join(args.output_dir, 'short_range/')
    )
    output_directory = os.path.join(args.output_dir,'long_range/') 
else:
    model = GPT2LMHeadModel(config = config)
    output_directory = os.path.join(args.output_dir,'short_range/')

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, 
    mlm = False, 
)

training_args = TrainingArguments(
    output_dir = output_directory, 
    overwrite_output_dir = True,
    learning_rate = args.learning_rate,
    adam_epsilon = args.adam_epsilon,
    adam_beta1 = args.adam_beta1,
    adam_beta2 = args.adam_beta2,
    weight_decay = args.weight_decay,
    warmup_steps = args.warmup_steps,
    num_train_epochs = args.num_train_epochs,
    per_device_train_batch_size = args.batch_size,
    save_steps = 500,
    save_total_limit = 1,
    do_eval = True,
    evaluation_strategy = 'epoch',
    seed = args.seed,
)
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = data_train,
    eval_dataset = data_eval,
    prediction_loss_only = True,
)

start = time.time()

trainer.train()

end = time.time()
write_time(start, end, output_directory)

trainer.save_model(output_directory)
