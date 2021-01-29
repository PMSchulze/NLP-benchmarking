
## BERT Final Validation Loss

Short range:
```
for VARIANT in 128_2_2_512_6 192_2_2_768_6 288_2_2_1152_6 384_2_2_1536_6 544_2_2_2176_6 128_5_2_512_6 \
128_10_2_512_6 128_18_2_512_6 128_36_2_512_6 204_7_2_816_6 256_9_2_1024_6 256_9_2_1024_3/32 256_9_2_1024_3/64
do
    python ./NLP-benchmarking/evaluation/evaluate_bert.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_nextsentence.txt \
        --block_size 128 \
        --model_name_or_path /home/ubuntu/lrz_share/models/bert/${VARIANT}/short_range/
done
```
Long range:
```
for VARIANT in 128_2_2_512_6 192_2_2_768_6 288_2_2_1152_6 384_2_2_1536_6 544_2_2_2176_6 128_5_2_512_6 \
128_10_2_512_6 128_18_2_512_6 128_36_2_512_6 204_7_2_816_6 256_9_2_1024_6 256_9_2_1024_3/32 256_9_2_1024_3/64
do
    python ./NLP-benchmarking/evaluation/evaluate_bert.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_nextsentence.txt \
        --block_size 512 \
        --model_name_or_path /home/ubuntu/lrz_share/models/bert/${VARIANT}/long_range/
done
```

## RoBERTa Final Validation Loss

Short range:
```
for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10 128_5_2_512_10 \
128_10_2_512_10 128_18_2_512_10 128_36_2_512_10 204_7_2_816_10 256_9_2_1024_10 256_9_2_1024_5/32 256_9_2_1024_5/64
do
    python ./NLP-benchmarking/evaluation/evaluate_roberta.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_linebyline.txt \
        --block_size 128 \
        --model_name_or_path ./models/roberta/${VARIANT}/short_range/
done
```
Long range:
```
for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10 128_5_2_512_10 \
128_10_2_512_10 128_18_2_512_10 128_36_2_512_10 204_7_2_816_10 256_9_2_1024_10 256_9_2_1024_5/32 256_9_2_1024_5/64
do
    python ./NLP-benchmarking/evaluation/evaluate_roberta.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_linebyline.txt \
        --block_size 512 \
        --model_name_or_path ./models/roberta/${VARIANT}/long_range/
done
```

## GPT-2 Final Validation Loss

Short range:
```
for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10 128_5_2_512_10 \
128_10_2_512_10 128_18_2_512_10 128_36_2_512_10 204_7_2_816_10 256_9_2_1024_10 256_9_2_1024_5/32 256_9_2_1024_5/64
do
    python ./NLP-benchmarking/evaluation/evaluate_gpt2.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_linebyline.txt \
        --block_size 128 \
        --model_name_or_path ./models/gpt2/${VARIANT}/short_range/
done
```
Long range:
```
for VARIANT in 128_2_2_512_10 192_2_2_768_10 288_2_2_1152_10 384_2_2_1536_10 544_2_2_2176_10 128_5_2_512_10 \
128_10_2_512_10 128_18_2_512_10 128_36_2_512_10 204_7_2_816_10 256_9_2_1024_10 256_9_2_1024_5/32 256_9_2_1024_5/64
do
    python ./NLP-benchmarking/evaluation/evaluate_gpt2.py \
        --corpus_eval ./data/pretrain_data/general/wiki_eval_linebyline.txt \
        --block_size 512 \
        --model_name_or_path ./models/gpt2/${VARIANT}/long_range/
done
```
