
## 0. Installation

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

Version: 3.4.0

## 1. Generation of Token Vocabulary

### 1.1. BERT
```
from tokenizers import BertWordPieceTokenizer

# Specify path of pre-training data
vocab_path = "/home/ubuntu/lrz_share/data/pretrain_data/source/wiki_train.txt"

# Initialize BERT's WordPiece tokenizer 
tokenizer = BertWordPieceTokenizer()

# Generate WordPiece token vocabulary from pre-training data
tokenizer.train(vocab_path)

# Save the vocabulary
tokenizer.save_model("/home/ubuntu/lrz_share/data/token_vocab/bert/")
```

### 1.2. GPT-2
```
from tokenizers import ByteLevelBPETokenizer

# Specify path of pre-training data
vocab_path = "/home/ubuntu/lrz_share/data/pretrain_data/source/wiki_train.txt"

# Initialize GPT's BPE tokenizer 
tokenizer = ByteLevelBPETokenizer()

# Generate BPE token vocabulary from pre-training data
tokenizer.train(
    files=vocab_path, 
    vocab_size=30_000, 
    min_frequency=2, 
    special_tokens=['<|endoftext|>', '<pad>']
)

# Save the vocabulary
tokenizer.save_model("/home/ubuntu/lrz_share/data/token_vocab/gpt2/")

```

### 1.3. RoBERTa
```
from tokenizers import ByteLevelBPETokenizer

# Specify path of pre-training data
vocab_path = "/home/ubuntu/lrz_share/data/pretrain_data/source/wiki_train.txt"

# Initialize RoBERTa's BPE tokenizer 
tokenizer = ByteLevelBPETokenizer()

# Generate BPE token vocabulary from pre-training data
tokenizer.train(
    files=vocab_path, 
    vocab_size=30_000, 
    min_frequency=2, 
    special_tokens=['<s>','<pad>','</s>','<mask>','<unk>']
)

# Save the vocabulary
tokenizer.save_model("/home/ubuntu/lrz_share/data/token_vocab/roberta/")
```
