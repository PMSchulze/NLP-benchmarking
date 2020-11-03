## Prepare data for huggingface's text processing classes

See https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py

```
from utils_data_preparation import (
    prepare_linebyline, 
    prepare_nextsentence,
    split_documents_by_len,
)
import os.path

datadir = '/home/ubuntu/lrz_share/data/pretrain_data'
```

We first put each document (i.e., each wikipedia section) on a single line 
and furthermore drop all documents with less than 20 characters:
```
prepare_linebyline(
    input_file_path = os.path.join(datadir, 'source/wiki_train.txt'), 
    output_file_path = os.path.join(datadir, 'general/wiki_train_linebyline.txt')
)
```

We then split the data into the p=0.9 shortest documents and the 1-p 
longest documents:
```
wiki_train_linebyline_short, wiki_train_linebyline_long = split_documents_by_len(
    input_file_path = os.path.join(datadir, 'general/wiki_train_linebyline.txt'),
    p = 0.9
)
```

### 1. Prepare for LineByLineTextDataset

Finally, we further divide each document into chunks of sentences.
On each line, we iteratively add consecutive sentences from a respective document
and stop after the total line length (i.e., number of characters) exceeds n. 
We also drop chunks with length<20 characters. For details, please check
the function 'divide_into_chunks' in [this](https://github.com/PMSchulze/masters_thesis/blob/master/data_preparation/utils_data_preparation.py) script.
We find that, on average, one BPE token corresponds to approximately 5 characters.
```
wiki_train_linebyline_128, wiki_train_linebyline_512 =  divide_into_chunks(
    input_file_short = wiki_train_linebyline_short,
    input_file_long = wiki_train_linebyline_long,
    len_short = 128*5,
    len_long = 512*5
)

len(wiki_train_linebyline_128), len(wiki_train_linebyline_512)
# (638590, 60850)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_512.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_512:
        print(line, file = text_file)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_128.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_128:
        print(line, file = text_file)
```

### 2. Prepare for TextDatasetForNextSentencePrediction

Apart from sampling random sentences for the NSP task, in contrast to LineByLineTextDataset, the class TextDatasetForNextSentencePrediction 
already fills the text chunks to the desired length (LineByLineTextDataset simply cuts off after the specified block_size). 
Therefore, we do not have to divide the text into smaller chunks manually.

In order to ensure that training of BERT (for which we use TextDatasetForNextSentencePrediction) is similar to training of RoBERTa
and GPT-2, we use the same portions of the data for short- and long-range dependencies ('wiki_train_linebyline_short' and 'wiki_train_linebyline_long', respectively).

```
from utils_data_preparation import prepare_nextsentence

wiki_train_linebyline_long = prepare_nextsentence(
    input_file = wiki_train_linebyline_long,
    output_file_path = os.path.join(datadir, 'general/wiki_train_nextsentence_long.txt')
)
wiki_train_linebyline_short = prepare_nextsentence(
    input_file = wiki_train_linebyline_short,
    output_file_path = os.path.join(datadir, 'general/wiki_train_nextsentence_short.txt')
)
```
