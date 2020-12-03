import argparse
import os.path
import pickle
from pretrain_utils import write_time
import time
import torch
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertTokenizerFast,
    DataCollatorForNextSentencePrediction,
    TextDatasetForNextSentencePrediction,
    Trainer,
    TrainingArguments
)

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type = int)
parser.add_argument("--num_hidden_layers", type = int)
parser.add_argument("--num_attention_heads", type = int)
parser.add_argument("--intermediate_size", type = int)
parser.add_argument("--num_train_epochs", type = float)
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
parser.add_argument("--evaluation_strategy", default = 'epoch') 
parser.add_argument("--eval_steps", type = float, default = None) 
parser.add_argument("--hidden_dropout_prob", type = float, default = 0.1) 
parser.add_argument("--learning_rate", type = float, default = 1e-4) 
parser.add_argument("--adam_epsilon", type = float, default = 1e-06) 
parser.add_argument("--adam_beta1", type = float, default = 0.9) 
parser.add_argument("--adam_beta2", type = float, default = 0.999) 
parser.add_argument("--weight_decay", type = float, default = 0.01) 
parser.add_argument("--long_range", type = bool, default = False) 

args = parser.parse_args()


tokenizer = BertTokenizerFast.from_pretrained(
    args.token_vocab,
)

data_train = TextDatasetForNextSentencePrediction(
    tokenizer = tokenizer,
    file_path = args.corpus_train, 
    block_size = args.block_size,
)

data_eval = TextDatasetForNextSentencePrediction(
    tokenizer = tokenizer,
    file_path = args.corpus_eval, 
    block_size = args.block_size,
)

config = BertConfig(
    vocab_size = len(tokenizer),
    hidden_size = args.hidden_size,
    num_hidden_layers = args.num_hidden_layers,
    num_attention_heads = args.num_attention_heads,
    intermediate_size = args.intermediate_size,
    attention_probs_dropout_prob = args.attention_probs_dropout_prob,
    hidden_dropout_prob = args.hidden_dropout_prob,
)

if args.long_range:
    model = BertForPreTraining.from_pretrained(
        os.path.join(args.output_dir, 'short_range/')
    )
    output_directory = os.path.join(args.output_dir,'long_range/') 
else:
    model = BertForPreTraining(config = config)
    output_directory = os.path.join(args.output_dir,'short_range/')

data_collator = DataCollatorForNextSentencePrediction(
    tokenizer = tokenizer, 
    mlm = True, 
    mlm_probability = 0.15,
    block_size = args.block_size
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
    evaluation_strategy = args.evaluation_strategy,
    eval_steps = args.eval_steps,
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
