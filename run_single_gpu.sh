#!/bin/bash

python main.py \
  --model_name convnext_xlarge \
  --batch_size 16 \
  --n_steps 100 \
  --save_model \
  --project_name chexpert \
  --run_name test \
  --data_dir /scratch/rawhad/CSE507/practice_1/chexpert_preprocessed_ds
