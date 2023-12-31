#!/bin/bash

deepspeed main.py \
   --data_path yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-6.7b \
   --per_device_train_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --zero_stage 3 \
   --gradient_checkpointing \
   --offload \
   --deepspeed \
   --output_dir ./output
