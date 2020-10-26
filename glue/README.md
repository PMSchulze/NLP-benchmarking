## Fine-tuning GPT2 on GLUE

```
pip install datasets
```

```
python /home/ubuntu/masters_thesis/glue/finetune_gpt2_glue.py \
  --batch_size 32 \
  --cache_dir /home/ubuntu/lrz_share/huggingface_datasets/ \
  --hidden_size 128 \
  --model_name_or_path /home/ubuntu/lrz_share/models/gpt2/128_2_2_512_10/ \
  --num_train_epochs 3 \
  --output_dir /home/ubuntu/lrz_share/fine_tuned/gpt2/glue \
  --seed 2020 \
  --task 'SST-2' \
  --token_vocab /home/ubuntu/data/token_vocab/gpt2
```


## Fine-tuning RoBERTa on GLUE

```
export GLUE_DIR=/home/ubuntu/data/glue
export MODEL=roberta
export SEED=2020

for VARIANT in 192_2_2_768_10 384_2_2_1536_10 544_2_2_2176_10 128_5_2_512_10 128_10_2_512_10 128_18_2_512_10 128_36_2_512_10
do
    cp /home/ubuntu/data/token_vocab/$MODEL/* /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/

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
