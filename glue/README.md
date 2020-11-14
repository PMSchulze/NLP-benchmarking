We report accuracy for all tasks execpt for CoLA (MCC), QQP (F1), MRPC (F1) and STS-B (Spearman's Corr).

## Fine-tuning BERT on GLUE

For BERT and RoBERTa we can use the script [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py) from the transformers library. 

Note that the shell script has to be run from the transformers repository.

```
export MODEL=bert
export SEED=2020

for VARIANT in 128_36_2_512_6
do
    cp /home/ubuntu/lrz_share/data/token_vocab/$MODEL/vocab.txt /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/long_range/vocab.txt

    for TASK in MNLI QQP QNLI RTE WNLI CoLA SST-2 MRPC STS-B
    do
        python /home/ubuntu/transformers/examples/text-classification/run_glue.py \
            --model_name_or_path /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/long_range/ \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --max_seq_length 512 \
            --per_device_train_batch_size 16   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/$MODEL/${VARIANT}/glue/${TASK}/ \
            --overwrite_output_dir \
            --seed $SEED
    done
done
```


## Fine-tuning RoBERTa on GLUE

Again, the shell script has to be run from the transformers repository.

```
export MODEL=roberta
export SEED=2020

for VARIANT in 128_5_2_512_10
do
    cp /home/ubuntu/lrz_share/data/token_vocab/$MODEL/* /home/ubuntu/lrz_share/models/$MODEL/${VARIANT}/long_range

    for TASK in SST2 QNLI RTE CoLA WNLI QQP MRPC STSB MNLI
    do
        python /home/ubuntu/transformers/examples/text-classification/run_glue.py \
            --model_name_or_path /home/ubuntu/lrz_share/models/short_range/$MODEL/${VARIANT}/long_range/ \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --max_seq_length 512 \
            --per_device_train_batch_size=8   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/$MODEL/glue/${VARIANT}/${TASK}/ \
            --overwrite_output_dir \
            --seed $SEED
    done
done
```

## Fine-tuning GPT-2 on GLUE

For GPT-2 we have to write [our own training script](https://github.com/PMSchulze/masters_thesis/blob/master/glue/finetune_gpt2_glue.py). We also have to [define classification heads and data preprocessing steps](https://github.com/PMSchulze/masters_thesis/blob/master/glue/utils_gpt2_glue.py). The data itself can be conveniently loaded using the [datasets library](https://huggingface.co/docs/datasets/).

```
pip install datasets
```

Run this script from masters_thesis/glue/ in this repository.

```
export SEED=2020

for VARIANT in 128_36_2_512_10
do
    for TASK in QQP
    do
        python /home/ubuntu/masters_thesis/glue/finetune_gpt2_glue.py \
            --batch_size 16 \
            --cache_dir /home/ubuntu/lrz_share/huggingface_datasets/ \
            --model_name_or_path /home/ubuntu/lrz_share/models/gpt2/${VARIANT}/long_range/ \
            --num_train_epochs 3 \
            --output_dir /home/ubuntu/lrz_share/fine_tuned/gpt2/glue/${VARIANT}/ \
            --seed $SEED \
            --task ${TASK} \
            --token_vocab /home/ubuntu/lrz_share/data/token_vocab/gpt2
    done
done
```

```
export SEED=2020

for VARIANT in 544_2_2_2176_10
do
    python /home/ubuntu/masters_thesis/glue/finetune_gpt2_mnli.py \
        --batch_size 16 \
        --cache_dir /home/ubuntu/lrz_share/huggingface_datasets/ \
        --model_name_or_path /home/ubuntu/lrz_share/models/gpt2/${VARIANT}/long_range/ \
        --num_train_epochs 3 \
        --output_dir /home/ubuntu/lrz_share/fine_tuned/gpt2/glue/${VARIANT}/ \
        --seed $SEED \
        --token_vocab /home/ubuntu/lrz_share/data/token_vocab/gpt2
done
```
