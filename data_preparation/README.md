## Prepare data for huggingface's text processing classes

See https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py

### Prepare for LineByLineTextDataset

We generate train and test data, where each text document is on a single line:
```
from utils_data_preparation import (
    prepare_linebyline, 
    prepare_linebyline_n, 
    split_documents_by_len
)
import os.path

datadir = '/home/ubuntu/lrz_share/data/pretrain_data'

# We first put each document (i.e., each wikipedia section) on a single line:
# Furthermore, we drop all documents with less than 20 characters.
prepare_linebyline(
    input_file = os.path.join(datadir, 'source/wiki_train.txt'), 
    output_file = os.path.join(datadir, 'general/wiki_train_linebyline.txt')
)
prepare_linebyline(
    input_file = os.path.join(datadir, 'source/wiki_test.txt'), 
    output_file = os.path.join(datadir, 'general/wiki_test_linebyline.txt')
)


wiki_train_linebyline_short, wiki_train_linebyline_long = split_documents_by_len(
    input_file = os.path.join(datadir, 'general/wiki_train_linebyline.txt'),
    p = 0.9
)

# We then further divide each document into chunks of sentences.
# On each line, we iteratively add consecutive sentences from a respective document
# and stop after the total line length (i.e., number of characters) exceeds n. 
# We find that, on average, one BPE token corresponds to approximately 5 characters. 

# Prepare for usage with LineByLineTextDataset with block_size 128 and BPE tokenizer
prepare_linebyline_n(
    input_file = wiki_train_linebyline_short, 
    output_file = os.path.join(datadir, 'general/wiki_train_linebyline_128.txt'),
    n = 128*5
)
prepare_linebyline_n(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline_128.txt',
    n = 128*5
)

# Prepare for usage with LineByLineTextDataset with block_size 512 and BPE tokenizer
prepare_linebyline_n(
    input_file = os.path.join(datadir, 'general/wiki_train_linebyline.txt'), 
    output_file = os.path.join(datadir, 'general/wiki_train_linebyline_512.txt'),
    n = 512*5
)
prepare_linebyline_n(
    input_file = wiki_train_linebyline_long, 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline_512.txt',
    n = 512*5
)
```

### Prepare for TextDatasetForNextSentencePrediction

We generate train and test data, where each sentence is on a single line and text documents are separated with a blank line:
```
from utils_data_preparation import prepare_nextsentence

prepare_nextsentence(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt',
    output_file = '/home/ubuntu/data/pretrain_data/wiki_train_nextsentence.txt'
)
prepare_nextsentence(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_nextsentence.txt'
)
```
