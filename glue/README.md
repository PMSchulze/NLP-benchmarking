We report accuracy for all tasks execpt for CoLA (MCC), QQP (F1), MRPC (F1) and STS-B (Spearman's Corr).

## Fine-tuning GPT2 on GLUE

```
pip install datasets
```

```
export SEED=2020

for VARIANT in 128_2_2_512_10
do
    for TASK in QQP
    do
        python /home/ubuntu/masters_thesis/glue/finetune_gpt2_glue.py \
            --batch_size 32 \
            --cache_dir /home/ubuntu/lrz_share/huggingface_datasets/ \
            --model_name_or_path /home/ubuntu/lrz_share/models/gpt2/${VARIANT}/ \
            --num_train_epochs 3 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/gpt2/glue/${VARIANT}/ \
            --seed $SEED \
            --task ${TASK} \
            --token_vocab /home/ubuntu/data/token_vocab/gpt2
    done
done
```


## Fine-tuning BERT on GLUE

For BERT and RoBERTa we can use the script [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py) from the transformers library. This script assumes that the GLUE data has been downloaded and stored:
`python utils/download_glue_data.py --data_dir ~/lrz_share/data/glue --tasks all`

```
export GLUE_DIR=/home/ubuntu/lrz_share/data/glue
export MODEL=bert
export SEED=2020

for VARIANT in 128_2_2_512_12 128_2_2_512_15 128_2_2_512_17
do
    cp /home/ubuntu/lrz_share/data/token_vocab/$MODEL/vocab.txt /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/vocab.txt

    for TASK in SST-2 QNLI RTE CoLA WNLI QQP MRPC STS-B MNLI
    do
        python /home/ubuntu/transformers/examples/text-classification/run_glue.py \
            --model_name_or_path /home/ubuntu/lrz_share/models/$MODEL/${VARIANT} \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --data_dir $GLUE_DIR/${TASK} \
            --max_seq_length 128 \
            --per_device_train_batch_size=32   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/$MODEL/${VARIANT}/glue/${TASK}/ \
            --overwrite_output_dir \
            --seed $SEED
    done
done
```


## Fine-tuning RoBERTa on GLUE

```
export GLUE_DIR=/home/ubuntu/lrz_share/data/glue
export MODEL=roberta
export SEED=2020

for VARIANT in 128_5_2_512_10
do
    cp /home/ubuntu/lrz_share/data/token_vocab/$MODEL/* /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/

    for TASK in SST-2 QNLI RTE CoLA WNLI QQP MRPC STS-B MNLI
    do
        python /home/ubuntu/transformers/examples/text-classification/run_glue.py \
            --model_name_or_path /home/ubuntu/lrz_share/models/$MODEL/${VARIANT} \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --data_dir $GLUE_DIR/${TASK} \
            --max_seq_length 128 \
            --per_device_train_batch_size=32   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/$MODEL/glue/${VARIANT}/${TASK}/ \
            --overwrite_output_dir \
            --seed $SEED
    done
done
```
