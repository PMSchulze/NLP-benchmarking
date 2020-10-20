## Fine-tuning GPT2 on GLUE

```
pip install datasets
```

```
python /home/ubuntu/masters_thesis/glue/finetune_new.py \
  --batch_size 32 \
  --cache_dir /home/ubuntu/lrz_share/huggingface_datasets/ \
  --hidden_size 128 \
  --model_name_or_path /home/ubuntu/lrz_share/models/gpt2/128_2_2_512_10/ \
  --num_train_epochs 3
  --output_dir /home/ubuntu/fine_tuned/gpt2 \
  --seed 2020 \
  --task 'CoLA' \
  --token_vocab /home/ubuntu/data/token_vocab/
```
