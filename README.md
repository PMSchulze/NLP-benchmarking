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

Colons can be used to align columns.

| Tables        | Are   |       | Cool  |       |       |       |       |       |       |       |
|------ |-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|       |       | $1600 |       |       |       |       |       |       |       |       |       |
|       |       |   $12 |       |       |       |       |       |       |       |       |       |
|       |       |    $1 |       |       |       |       |       |       |       |       |       |


## Fine-tuning
