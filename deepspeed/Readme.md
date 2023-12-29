# docker environment
docker run -itd --shm-size 64g --gpus all --name deepspeed_test nvidia/cuda:12.1.0-devel-ubuntu20.04
docker exec -it deepspeed_test bash

# deepspeed runtime environment preparation
apt update && apt install vim python3 git -y
apt install python-is-python3 pip -y
pip install git+https://github.com/huggingface/transformers
