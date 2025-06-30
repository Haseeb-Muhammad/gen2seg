#!/bin/bash

accelerate launch --multi_gpu /netscratch/muhammad/codes/gen2seg/training/train_SD_Fluo.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
  --max_train_steps 30000 \
  --checkpointing_steps 10000 \
  --train_batch_size 2  \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 6e-05 \
  --lr_total_iter_length 40000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "/netscratch/muhammad/codes/gen2seg/training/model-finetuned/stable_diffusion_fluo" \
  --enable_xformers_memory_efficient_attention \
  "$@"