# Benchmarking down-scaled (not so large) pre-trained language models

This is a benchmarking study, which aims to systematically  compare  the  pre-training  objectives of three popular NLP systems, BERT ([Devlin et. al, 2019](https://arxiv.org/abs/1810.04805)), RoBERTa ([Liu et. al, 2019](https://arxiv.org/abs/1907.11692)) and GPT-2 ([Radford et. al, 2018](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)), for different shape parameters and model sizes,  while  also  varying  the  number  of  pre-training steps and the batch size.

## Preliminaries

### Installation

As a first step, please install HuggingFace's [transformers](https://github.com/huggingface/transformers) library ([Wolf et. al, 2020](https://arxiv.org/abs/1910.03771)) from source. 

You need to install the specific version that was used to run our experiments. Therefore, fork the [transformers](https://github.com/huggingface/transformers) repository and create a local clone of the fork in the root directory of this project. Then run:

```
cd ./transformers
git reset --hard 47dfa65b0cba3d4fb3f24e52bc2299e261119276
git push -f origin master
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

First, we create the token vocabulary from WikiText-103 ([Merity et. al, 2016](https://arxiv.org/abs/1609.07843)), which can be downloaded from [here](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) (raw / character-level version). We store the data in data/pretrain_data/source/ with filenames wiki_train.txt, wiki_test.txt and wiki_eval.txt.

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

How fine-tuning on GLUE ([Wang et. al, 2018](https://arxiv.org/abs/1804.07461)) can be performed is explained [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/glue).

### 5. Obtain Validation Loss

Finally, [here](https://github.com/PMSchulze/NLP-benchmarking/tree/master/evaluation) we have written scripts to conveniently calculate the validation losses for the different systems.

### 6. Gather Results

If you have access to our results, [these](https://github.com/PMSchulze/NLP-benchmarking/tree/master/results) R files can be used to gather all results conveniently. 
