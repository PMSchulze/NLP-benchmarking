## RoBERTa

```
export DATA_DIR=/home/ubuntu/lrz_share/data/
export OUTPUT_DIR=/home/ubuntu/lrz_share/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10
do
    python /home/ubuntu/masters_thesis/pretraining/pretrain_roberta.py \
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

```
export DATA_DIR=/home/ubuntu/lrz_share/data/
export OUTPUT_DIR=/home/ubuntu/lrz_share/models/

for VARIANT in 128_2_2_512_10
do
    python /home/ubuntu/masters_thesis/pretraining/pretrain_roberta.py \
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

## BERT

```
export DATA_DIR=/home/ubuntu/lrz_share/data/
export OUTPUT_DIR=/home/ubuntu/lrz_share/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10
do
    python /home/ubuntu/masters_thesis/pretraining/pretrain_bert.py \
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

```
export DATA_DIR=/home/ubuntu/lrz_share/data/
export OUTPUT_DIR=/home/ubuntu/lrz_share/models/

for VARIANT in 128_2_2_512_10 128_5_2_512_10 128_10_2_512_10
do
    python /home/ubuntu/masters_thesis/pretraining/pretrain_bert.py \
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
