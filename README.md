## Generation of Token Vocabulary

```
from tokenizers import BertWordPieceTokenizer

# Specify path of pre-training data
vocab_path = "/home/ubuntu/data/pretrain_data/wiki_train.txt"

# Initialize BERT's WordPiece tokenizer 
tokenizer = BertWordPieceTokenizer()

# Generate WordPiece token vocabulary from pre-training data
tokenizer.train(vocab_path)

# Save the vocabulary
tokenizer.save_model("/home/ubuntu/data/token_vocab/bert/")
```

## Pre-training

### Bert with half-sized architecture components
```
python ~/python_files/pretrain_bert.py \
    --hidden_size 384 \
    --num_hidden_layers 6 \
    --num_attention_heads 6 \
    --intermediate_size 1536 \
    --num_train_epochs 10 \
    --output_dir /home/ubuntu/models/bert/bert_half \
    --corpus_pretrain /home/ubuntu/data/pretrain_data/wiki_train.txt \
    --token_vocab /home/ubuntu/data/token_vocab/bert/
```


Hyperparameter                | Specification
------------------------------| -----------------
num_hidden_layers             :|: 6
num_attention_heads           | 6
intermediate_size             | 1536
num_train_epochs              | 10
attention_probs_dropout_prob  | 0.1
hidden_dropout_prob           | 0.1
block_size                    | 128
learning_rate                 | 1e-4
weight_decay                  | 0.01
warmup_steps                  | 1820
per_gpu_train_batch_size      | 64
save_steps                    | 10_000
save_total_limit              | 2
 

## Fine-tuning
