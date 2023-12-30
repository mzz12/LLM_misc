#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
mkdir -p ./output
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12345 \
    run_clm_no_trainer.py \
    --random_ltd \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2-medium \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --deepspeed_config config.json \
    --deepspeed \
    --output_dir ./output
