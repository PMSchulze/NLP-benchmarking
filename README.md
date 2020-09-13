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

```
lalala
```

## Fine-tuning
