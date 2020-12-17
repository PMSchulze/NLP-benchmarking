
## 0. Installation

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

Version: 3.4.0

## 1. Generation of Token Vocabulary

First, we create the token vocabulary from WikiText-103. Please adjust the paths to your own folders.

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

## 2. Data Preparation

Next, we prepare the data, such that it can be processed by the different systems. This is done [here](https://github.com/PMSchulze/masters_thesis/tree/master/data_preparation).

## 3. Pre-Training

For pre-training, we have written a training script for each system. 

To reproduce our experiments, run the scripts from the command line with the arguments specified [here](https://github.com/PMSchulze/masters_thesis/tree/master/pretraining).

## 4. Fine-Tuning on GLUE

How fine-tuning on GLUE can be performed is explained [here](https://github.com/PMSchulze/masters_thesis/tree/master/glue).

## 5. Obtain Validation Loss

Finally, [here](https://github.com/PMSchulze/masters_thesis/tree/master/evaluation) we have written scripts to conveniently calculate the validation losses for the different systems.

## 6. Gather Results
