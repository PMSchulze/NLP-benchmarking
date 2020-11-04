## Prepare data for huggingface's text processing classes

See https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py

Load utility functions and specify directory of pretraining data.
```
from utils_data_preparation import prepare_linebyline, split_documents_by_len
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

### 1. Prepare for TextDatasetForNextSentencePrediction

Apart from sampling random sentences for the NSP task, in contrast to *LineByLineTextDataset*, the class *TextDatasetForNextSentencePrediction* 
already fills the text chunks to the desired length. Therefore, we do not have to divide the text into smaller chunks manually.

In order to ensure that training of *BERT* (for which we use *TextDatasetForNextSentencePrediction*) is similar to training of *RoBERTa*
and *GPT-2*, we use the same portions of the data for short- and long-range dependencies (corresponding to the previously generated  *wiki_train_linebyline_short* and *wiki_train_linebyline_long*, respectively).

The only step that we perform in the following is to put each sentence of a document on a separate line and separate documents with a blank line (we also, as above, drop sentences with length<20 characters); this is the expected format of *TextDatasetForNextSentencePrediction*. For *wiki_train_nextsentence_long.txt* we will then specify *block_size=512* and for *wiki_train_nextsentence_short.txt* we set *block_size=128* when instantiating the object of type *TextDatasetForNextSentencePrediction* (this occurs directly before we start pretraining *BERT*, i.e., in the pretraining script).

```
from utils_data_preparation import prepare_nextsentence

prepare_nextsentence(
    input_file = wiki_train_linebyline_long,
    output_file_path = os.path.join(datadir, 'general/wiki_train_nextsentence_long.txt')
)
prepare_nextsentence(
    input_file = wiki_train_linebyline_short,
    output_file_path = os.path.join(datadir, 'general/wiki_train_nextsentence_short.txt')
)
```


### 2. Prepare for LineByLineTextDataset

In order to prepare the data for the class *LineByLineTextDataset*, we have to divide each document into chunks of sentences, because *LineByLineTextDataset* simply cuts off after the specified length (which is set with the *block_size* argument). 

On each line, we therefore iteratively add consecutive sentences from a respective document
and stop after the total line length (i.e., number of characters) exceeds n. 
For details, please check the function *divide_into_chunks* [in this script](https://github.com/PMSchulze/masters_thesis/blob/master/data_preparation/utils_data_preparation.py).
We find that, on average, one BPE token corresponds to approximately 5 characters.
```
from utils_data_preparation import divide_into_chunks

wiki_train_linebyline_128, wiki_train_linebyline_512 =  divide_into_chunks(
    input_file_short = os.path.join(datadir, 'general/wiki_train_nextsentence_short.txt'),
    input_file_long = os.path.join(datadir, 'general/wiki_train_nextsentence_long.txt'),
    len_short = 128*5,
    len_long = 512*5
)

len(wiki_train_linebyline_128), len(wiki_train_linebyline_512)
# (638590, 60850)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_long.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_512:
        print(line, file = text_file)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_short.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_128:
        print(line, file = text_file)
```
