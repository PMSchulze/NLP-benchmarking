## Prepare data for huggingface's text processing classes

### Prepare for LineByLineTextDataset

We generate train and test data, where each text document is on a single line:
```
from utils_data_preparation import prepare_linebyline

prepare_linebyline('/home/ubuntu/data/pretrain_data/wiki_train.txt', '/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt')
prepare_linebyline('/home/ubuntu/data/pretrain_data/wiki_test.txt', '/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt')
```

### Prepare for TextDatasetForNextSentencePrediction

Generate train and test data, where each sentence is on a single line and documents are separated with a blank line:
```
from utils_data_preparation import prepare_nextsentence

prepare_nextsentence('/home/ubuntu/data/pretrain_data/wiki_train_linebyline.txt', '/home/ubuntu/data/pretrain_data/wiki_train_nextsentence.txt')
prepare_nextsentence('/home/ubuntu/data/pretrain_data/wiki_test_linebyline.txt', '/home/ubuntu/data/pretrain_data/wiki_test_nextsentence.txt')
```
