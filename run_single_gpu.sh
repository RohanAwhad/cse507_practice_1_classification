#!/bin/bash

python main.py \
  --model_name convnext_xlarge \
  --batch_size 128 \
  --n_steps 5240 \
  --save_model \
  --project_name cse507_practice_1_chexpert \
  --run_name chexpert_convnext_full_v1_run_1 \
  --data_dir /scratch/rawhad/CSE507/practice_1/chexpert_preprocessed_ds \
  --prefetch_size 4 \
  --use_worker \
  --dropout_rate 0.1 \
  --lr 3e-4 \
;
