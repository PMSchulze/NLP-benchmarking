## RoBERTa

```
export DATA_DIR=/home/ubuntu/lrz_share/data/
export OUTPUT_DIR=/home/ubuntu/lrz_share/models/

for VARIANT in 384_6_6_1536_10
do
    python /home/ubuntu/masters_thesis/pretraining/pretrain_roberta.py
        --hidden_size 384 \
        --num_hidden_layers 6 \
        --num_attention_heads 6 \
        --intermediate_size 1536 \
        --num_train_epochs 10 \
        --block_size 128 \
        --batch_size 64 \
        --warmup_steps 992 \
        --corpus_train ${DATA_DIR}pretrain_data/general/wiki_train_linebyline_short.txt \
        --corpus_eval ${DATA_DIR}pretrain_data/general/wiki_eval_linebyline.txt \
        --output_dir ${OUTPUT_DIR}roberta/${VARIANT}/ \
        --token_vocab ${DATA_DIR}token_vocab/roberta/
done
```

## BERT
