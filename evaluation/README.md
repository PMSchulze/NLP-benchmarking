```
for VARIANT in 128_2_2_512_6 192_2_2_768_6 288_2_2_1152_6 384_2_2_1536_6 544_2_2_2176_6 128_5_2_512_6 \
128_10_2_512_6 128_18_2_512_6 128_36_2_512_6 204_7_2_816_6 256_9_2_1024_6 256_9_2_1024_6/32/ 256_9_2_1024_6/64/
do
    python /home/ubuntu/masters_thesis/evaluation/evaluate_bert.py \
        --corpus_eval /home/ubuntu/lrz_share/data/pretrain_data/general/wiki_eval_nextsentence.txt \
        --block_size 128 \
        --model_name_or_path /home/ubuntu/lrz_share/models/bert/${VARIANT}/short_range/
done
```
