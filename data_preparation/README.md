## Prepare data for huggingface's text processing classes

See https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py

### Prepare for LineByLineTextDataset

We generate train and test data, where each text document is on a single line:
```
from utils_data_preparation import prepare_linebyline, prepare_linebyline_n

# We first put each document (i.e., each wikipedia section) on a single line: 
prepare_linebyline(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_train.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt'
)
prepare_linebyline(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_test.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt'
)


# We then further divide each document into chunks of sentences.
# On each line, we iteratively add consecutive sentences from a respective document
# and stop after the total line length exceeds n. This prepares the data for usage 
# of LineByLineTextDataset with blocksize n.
prepare_linebyline_n(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline_128.txt',
    n = 128
)
prepare_linebyline_n(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline_128.txt',
    n = 128
)

prepare_linebyline_n(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_train_linebyline_256.txt',
    n = 256
)
prepare_linebyline_n(
    input_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt', 
    output_file = '/home/ubuntu/data/pretrain_data/wiki_test_linebyline_256.txt',
    n = 256
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
