## 1 RoBERTa

### 1.1. Scaling Width

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17
done
```
Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17 \
        --long_range True
done
```

### 1.2 Scaling Depth

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10 128_18_2_512_10 128_36_2_512_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17
done
```
Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10 128_18_2_512_10 128_36_2_512_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17 \
        --long_range True
done
```

### 1.3 Scaling Width & Depth

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17
done
```
Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17 \
        --long_range True
done
```

### 1.4 Batch Size

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_5

for BATCHSIZE in 32 64
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size ${BATCHSIZE} \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/${BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17
done
```
Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_5

for BATCHSIZE in 32 64
do
    BATCHSIZE_LONG=$(($BATCHSIZE/4))
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size $BATCHSIZE_LONG \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/${BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17 \
        --long_range True
done
```

### 1.5 Attention Heads

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17
done
```
Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_roberta.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/ \
        --seed 17 \
        --long_range True
done
```

## 2 BERT

### 2.1 Scaling Width

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_6 192_2_2_768_6 288_2_2_1152_6 384_2_2_1536_6 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_6 192_2_2_768_6 288_2_2_1152_6 384_2_2_1536_6 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```
### 2.2 Scaling Depth

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_6 128_5_2_512_6 128_10_2_512_6 128_18_2_512_6 128_36_2_512_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_6 128_5_2_512_6 128_10_2_512_6 128_18_2_512_6 128_36_2_512_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```

### 2.3 Scaling Width & Depth

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_6 256_9_2_1024_6 256_9_4_1024_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_6 256_9_2_1024_6 256_9_4_1024_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```

### 2.4 Batch Size

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_3

for BATCHSIZE in 32 64
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size ${BATCHSIZE} \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/${BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_3

for BATCHSIZE in 32 64
do
    BATCHSIZE_LONG=$(($BATCHSIZE/4))
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size $BATCHSIZE_LONG \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/{$BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```

### 2.5 Grid Search

```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_6 104_3_2_416_6 90_4_2_360_6 74_6_2_296_6 64_8_2_256_6 58_10_2_232_6 52_12_2_208_6 48_14_2_192_6 46_16_2_184_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/grid_search/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --evaluation_strategy steps \
        --eval_steps 94000
done
```

### 2.6 Attention Heads

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_6
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```

### 2.7 Systematic Scaling

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 469_4_7_1876_5
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 469_4_7_1876_5
do
    python ./NLP-benchmarking/pretraining/pretrain_bert.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_nextsentence_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_nextsentence.txt \
        --output_dir ${OUTPUT_DIR}bert/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/bert/ \
        --seed 17 \
        --long_range True
done
```

## 3. GPT2

### 3.1 Scaling Width

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17 \
        --long_range True
done
```

### 3.1 Scaling Depth

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10 128_18_2_512_10 128_36_2_512_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10 128_18_2_512_10 128_36_2_512_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17 \
        --long_range True
done
```

### 3.3 Scaling Width & Depth

Short sequences:

```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17
done
```

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17 \
        --long_range True
done
```

### 3.4 Batch Size

Short sequences:

```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_5

for BATCHSIZE in 32 64
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size ${BATCHSIZE} \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/${BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17
done
```

Long sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/
export VARIANT=256_9_2_1024_5

for BATCHSIZE in 32 64
do
    BATCHSIZE_LONG=$(($BATCHSIZE/4))
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size $BATCHSIZE_LONG \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/${BATCHSIZE}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17 \
        --long_range True
done
```

### 3.5 Attention Heads

Short sequences:

```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 1000 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17
done
```

Short sequences:
```
export ROOT_DIR=$(pwd)
export DATA_DIR=$ROOT_DIR/data/
export OUTPUT_DIR=$ROOT_DIR/models/

for VARIANT in 544_2_8_2176_10
do
    python ./NLP-benchmarking/pretraining/pretrain_gpt2.py \
        --hidden_size $(echo $VARIANT| cut -d'_' -f 1) \
        --num_hidden_layers $(echo $VARIANT| cut -d'_' -f 2) \
        --num_attention_heads $(echo $VARIANT| cut -d'_' -f 3) \
        --intermediate_size $(echo $VARIANT| cut -d'_' -f 4) \
        --num_train_epochs $(echo $VARIANT | cut -d'_' -f 5) \
        --block_size 512 \
        --batch_size 16 \
        --warmup_steps 0 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_long.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}gpt2/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/gpt2/ \
        --seed 17 \
        --long_range True
done
```
