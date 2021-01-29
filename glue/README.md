We report accuracy for all tasks execpt for CoLA (MCC), QQP (F1), MRPC (F1) and STS-B (Spearman's Corr).

## Fine-tuning BERT on GLUE

For BERT and RoBERTa we can use the script [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py) from the transformers library. 

Note that the shell script has to be run from the transformers repository.

```
export MODEL=bert
export SEED=2020

for VARIANT in 204_7_2_816_6 256_9_2_1024_6
do
    cp ../data/token_vocab/$MODEL/vocab.txt ../models/$MODEL/${VARIANT}/long_range/vocab.txt

    for TASK in MNLI QQP QNLI RTE WNLI CoLA SST2 MRPC STSB
    do
        python ./examples/text-classification/run_glue.py \
            --model_name_or_path ../models/$MODEL/${VARIANT}/long_range/ \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --max_seq_length 512 \
            --per_device_train_batch_size 16   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir ../fine_tuned/$MODEL/${VARIANT}/glue/${TASK}/ \
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

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    cp ../data/token_vocab/$MODEL/* ../models/$MODEL/${VARIANT}/long_range

    for TASK in MNLI QQP QNLI RTE WNLI CoLA SST2 MRPC STSB
    do
        python ./examples/text-classification/run_glue.py \
            --model_name_or_path ../models/$MODEL/${VARIANT}/long_range/ \
            --task_name ${TASK} \
            --save_total_limit 1\
            --do_train \
            --do_eval \
            --max_seq_length 510 \
            --per_device_train_batch_size 16   \
            --learning_rate 2e-5 \
            --num_train_epochs 3.0 \
            --output_dir ../fine_tuned/$MODEL/${VARIANT}/glue/${TASK}/ \
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

```
export SEED=2020

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    for TASK in QQP QNLI RTE WNLI CoLA SST2 MRPC STSB
    do
        python ./NLP-benchmarking/glue/finetune_gpt2_glue.py \
            --batch_size 16 \
            --cache_dir ./huggingface_datasets/ \
            --model_name_or_path ./models/gpt2/${VARIANT}/long_range/ \
            --num_train_epochs 3 \
            --output_dir ./fine_tuned/gpt2/${VARIANT}/glue/ \
            --seed $SEED \
            --task ${TASK} \
            --token_vocab ./data/token_vocab/gpt2
    done
done
```

```
export SEED=2020

for VARIANT in 204_7_2_816_10 256_9_2_1024_10
do
    python ./NLP-benchmarking/glue/finetune_gpt2_mnli.py \
        --batch_size 16 \
        --cache_dir ./huggingface_datasets/ \
        --model_name_or_path ./models/gpt2/${VARIANT}/long_range/ \
        --num_train_epochs 3 \
        --output_dir ./fine_tuned/gpt2/${VARIANT}/glue/ \
        --seed $SEED \
        --token_vocab ./data/token_vocab/gpt2
done
```
