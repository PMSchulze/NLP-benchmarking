# Prepare data for huggingface's text processing classes

See https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py

## Prepare training data

Load utility functions and specify directory of pretraining data.
```
from utils_data_preparation import prepare_linebyline, split_documents_by_len
import os.path

# Original file wiki_train.txt should be in datadir/source/
datadir = '/home/ubuntu/lrz_share/data/pretrain_data'

# Files generated in this section are store in datadir/general/
if not os.path.exists('/home/ubuntu/lrz_share/data/pretrain_data/general'):
    os.makedirs('/home/ubuntu/lrz_share/data/pretrain_data/general')
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

len(wiki_train_linebyline_short), len(wiki_train_linebyline_long)
# (240327, 26703)
```

### 1. Prepare for TextDatasetForNextSentencePrediction

Apart from sampling random sentences for the NSP task, in contrast to *LineByLineTextDataset*, the class *TextDatasetForNextSentencePrediction* 
already fills the text chunks to the desired length. Therefore, we do not have to divide the text into smaller chunks manually.

In order to ensure that training of *BERT* (for which we use *TextDatasetForNextSentencePrediction*) is similar to training of *RoBERTa*
and *GPT-2*, for all models, we use the same portions of the data for learning short- and long-range dependencies during pretraining (corresponding to the previously generated  *wiki_train_linebyline_short* and *wiki_train_linebyline_long*, respectively).

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
    input_file_path_short = os.path.join(datadir, 'general/wiki_train_nextsentence_short.txt'),
    input_file_path_long = os.path.join(datadir, 'general/wiki_train_nextsentence_long.txt'),
    len_short = 128*5,
    len_long = 512*5
)

len(wiki_train_linebyline_128), len(wiki_train_linebyline_512)
# (634802, 60591)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_long.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_512:
        print(line, file = text_file)

with open(os.path.join(datadir, 'general/wiki_train_linebyline_short.txt'), 'w') as text_file:
    for line in wiki_train_linebyline_128:
        print(line, file = text_file)
```

## Prepare eval data

```
from utils_data_preparation import prepare_linebyline, split_documents_by_len
import os.path

# Original file wiki_train.txt should be in datadir/source/
datadir = '/home/ubuntu/lrz_share/data/pretrain_data'

# Files generated in this section are store in datadir/general/
if not os.path.exists('/home/ubuntu/lrz_share/data/pretrain_data/general'):
    os.makedirs('/home/ubuntu/lrz_share/data/pretrain_data/general')

prepare_linebyline(
    input_file_path = os.path.join(datadir, 'source/wiki_eval.txt'), 
    output_file_path = os.path.join(datadir, 'general/wiki_eval_linebyline.txt')
)

wiki_eval_linebyline, empty_dataset = split_documents_by_len(
    input_file_path = os.path.join(datadir, 'general/wiki_eval_linebyline.txt'),
    p = 1
)
len(wiki_eval_linebyline), len(empty_dataset)
#(540, 0)

from utils_data_preparation import prepare_nextsentence
prepare_nextsentence(
    input_file = wiki_eval_linebyline,
    output_file_path = os.path.join(datadir, 'general/wiki_eval_nextsentence.txt')
)
```

# Count corpus lengths to set training steps of BERT

### Short range 
```
import pickle
from transformers import RobertaTokenizerFast, BertTokenizerFast

with open('cached_lbl_RobertaTokenizerFast_128_wiki_train_linebyline_short.txt', 'rb') as f:
    data_roberta = pickle.load(f)

with open('cached_nsp_BertTokenizerFast_128_wiki_train_nextsentence_short.txt', 'rb') as f:
    data_bert = pickle.load(f)

tokenizer_roberta = RobertaTokenizerFast.from_pretrained('/home/ubuntu/lrz_share/data/token_vocab/roberta/')
tokenizer_bert = BertTokenizerFast.from_pretrained('/home/ubuntu/lrz_share/data/token_vocab/bert/')


len_roberta = 0

# for RoBERTa, we have to cut off the added start and end tokens
for i in range(len(data_roberta)):
    len_roberta += len(tokenizer_roberta.decode(data_roberta[i]['input_ids'][1:-1]))


len_bert = 0

for i in range(len(data_bert)):
    len_bert += len(tokenizer_bert.decode(data_bert[i]['tokens_a'])) + len(tokenizer_bert.decode(data_bert[i]['tokens_b']))

len_roberta, len_bert
# (308129512, 560497217)

factor_reduce_steps = len_roberta/len_bert

factor_reduce_steps
# 0.5497431613474005
```

### Long range 
```
import pickle
from transformers import RobertaTokenizerFast, BertTokenizerFast

with open('cached_lbl_RobertaTokenizerFast_512_wiki_train_linebyline_long.txt', 'rb') as f:
    data_roberta = pickle.load(f)

with open('cached_nsp_BertTokenizerFast_512_wiki_train_nextsentence_long.txt', 'rb') as f:
    data_bert = pickle.load(f)

tokenizer_roberta = RobertaTokenizerFast.from_pretrained('/home/ubuntu/lrz_share/data/token_vocab/roberta/')
tokenizer_bert = BertTokenizerFast.from_pretrained('/home/ubuntu/lrz_share/data/token_vocab/bert/')


len_roberta = 0

# for RoBERTa, we have to cut off the added start and end tokens
for i in range(len(data_roberta)):
    len_roberta += len(tokenizer_roberta.decode(data_roberta[i]['input_ids'][1:-1]))


len_bert = 0

for i in range(len(data_bert)):
    len_bert += len(tokenizer_bert.decode(data_bert[i]['tokens_a'])) + len(tokenizer_bert.decode(data_bert[i]['tokens_b']))

len_roberta, len_bert
#(124774651, 208814615)

factor_reduce_steps = len_roberta/len_bert

factor_reduce_steps
# 0.5975379213758577
```

