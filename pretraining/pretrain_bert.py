import argparse
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
import pickle
  
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

parser.add_argument(
    "--attention_probs_dropout_prob", type = float, default = 0.1
)
parser.add_argument("--hidden_dropout_prob", type = float, default = 0.1) 
parser.add_argument("--learning_rate", type = float, default = 1e-4) 
parser.add_argument("--adam_epsilon", type = float, default = 1e-06) 
parser.add_argument("--adam_beta1", type = float, default = 0.9) 
parser.add_argument("--adam_beta2", type = float, default = 0.999) 
parser.add_argument("--weight_decay", type = float, default = 0.01) 

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

model = BertForPreTraining(config = config)

data_collator = DataCollatorForNextSentencePrediction(
    tokenizer = tokenizer, 
    mlm = True, 
    mlm_probability = 0.15
)

training_args = TrainingArguments(
    output_dir = args.output_dir, 
    overwrite_output_dir = True,
    learning_rate = args.learning_rate,
    adam_epsilon = args.adam_epsilon,
    adam_beta1 = args.adam_beta1,
    adam_beta2 = args.adam_beta2,
    weight_decay = args.weight_decay,
    warmup_steps = args.warmup_steps,
    num_train_epochs = args.num_train_epochs,
    per_device_train_batch_size = args.batch_size,
    save_steps = 10_000,
    save_total_limit = 1,
    do_eval = True,
    evaluation_strategy = 'epoch',
)
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = data_train,
    eval_dataset = data_eval,
    # prediction_loss_only = True,
)

trainer.train()

trainer.save_model(args.output_dir)
