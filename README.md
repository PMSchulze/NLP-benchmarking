# Benchmarking down-scaled (not so large) pre-trained language models

## Description

In this benchmarking study, we systematically  compare  the  pre-training  objectives of three popular NLP systems, BERT, RoBERTa and GPT-2, for different shape parameters and model sizes,  while  also  varying  the  number  of  pre-training steps and the batch size

## Preliminaries

### Installation

Before we start, we have to install version 3.4.0 of huggingface's transformers library from source.

Therefore, go to https://github.com/huggingface/transformers/releases and download the source code of v3.4.0 to the root directory of this project.

Then, go to the root directory and run:

```
unzip transformers-3.4.0.zip -d transformers
cd transformers
pip install -e .
```

### Folder Structure

We assume the following folder structure for this project:
```
    .
    ├── NLP-benchmarking        # Local copy of this GitHub repository
    ├── transformers            # Local copy of huggingface/transformers repository
    ├── data                    # Data directory
    │   ├── pretrain_data       # Pre-training data
    │   │   ├── general         # Prepared data that is used as input of models
    │   │   └── source          # Unprepared source data 
    │   └── token_vocab         # Token vocabularies of different models
    │       ├── bert            #
    │       ├── gpt2            #
    │       └── roberta         #
    ├── models                  # Pre-trained models
    │   ├── bert                #
    │   ├── gpt2                # 
    │   └── roberta             # 
    └── fine_tuned              # Fine-tuned models
        ├── bert                # 
        ├── gpt2                # 
        └── roberta             # 
```

## Rerun Analysis

In order to rerun our experiments, the following steps have to be performed in the order given.

### 1. Generation of Token Vocabulary

First, we create the token vocabulary from WikiText-103, which can be downloaded from [here](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) (raw / character-level version). Please adjust the paths to your own folders.

#### 1.1. BERT
```
from tokenizers import BertWordPieceTokenizer

# Specify path of pre-training data
vocab_path = "data/pretrain_data/source/wiki_train.txt"

# Initialize BERT's WordPiece tokenizer 
tokenizer = BertWordPieceTokenizer()

# Generate WordPiece token vocabulary from pre-training data
tokenizer.train(vocab_path)

# Save the vocabulary
tokenizer.save_model("data/token_vocab/bert/")
```

#### 1.2. GPT-2
```
from tokenizers import ByteLevelBPETokenizer

# Specify path of pre-training data
vocab_path = "data/pretrain_data/source/wiki_train.txt"

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
tokenizer.save_model("data/token_vocab/gpt2/")

```

#### 1.3. RoBERTa
```
from tokenizers import ByteLevelBPETokenizer

# Specify path of pre-training data
vocab_path = "data/pretrain_data/source/wiki_train.txt"

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
tokenizer.save_model("data/token_vocab/roberta/")
```

### 2. Data Preparation

Next, we prepare the data, such that it can be processed by the different systems. This is done [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/data_preparation).

### 3. Pre-Training

For pre-training, we have written a training script for each system. 

To reproduce our experiments, run the scripts from the command line with the arguments specified [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/pretraining).

### 4. Fine-Tuning on GLUE

How fine-tuning on GLUE can be performed is explained [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/glue).

### 5. Obtain Validation Loss

Finally, [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/evaluation) we have written scripts to conveniently calculate the validation losses for the different systems.

### 6. Gather Results

If you have access to our results, [these](https://github.com/PMSchulze/NLP-benchmarking/tree/master/results) R files can be used to gather all results conveniently. 
