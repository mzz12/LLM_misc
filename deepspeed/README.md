# docker environment
docker run -itd --shm-size 64g --gpus all --name deepspeed_test nvidia/cuda:12.1.0-devel-ubuntu20.04
docker exec -it deepspeed_test bash

# deepspeed runtime environment preparation
apt update && apt install vim python3 git -y
apt install python-is-python3 pip -y
pip install git+https://github.com/huggingface/transformers
pip install transformers[deepspeed] datasets

# fetch this repository
git clone https://github.com/mzz12/LLM_misc.git
cd LLM_misc/deepspeed

# Lauching training
+ GPT2: accelerate launch ./deepspeed_with_config_support.py --model_type "gpt2" --tokenizer_name "gpt2" --dataset_name "wikitext" --dataset_config_name "wikitext-2-raw-v1" --block_size 128 --output_dir "./output" --learning_rate 5e-4 --per_device_train_batch_size 24 --num_train_epochs 1 --with_tracking
+ GPT2-xl (1.6B): accelerate launch ./deepspeed_with_config_support.py --model_name_or_path "gpt2-xl" --tokenizer_name "gpt2-xl" --dataset_name "wikitext" --dataset_config_name "wikitext-2-raw-v1" --block_size 128 --output_dir "./output" --learning_rate 5e-4 --per_device_train_batch_size 24 --num_train_epochs 1 --with_tracking