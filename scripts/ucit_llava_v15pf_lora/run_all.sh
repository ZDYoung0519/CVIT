
export NCCL_P2P_LEVEL=NVL
export BNB_CUDA_VERSION=124
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

CONFIG=cltuner/configs/ucit_llava_v15pf_lora.py
WORKDIR=work_dirs/ucit_llava_v15pf_lora


bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 0 $CONFIG $WORKDIR

bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 1 $CONFIG $WORKDIR

bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 2 $CONFIG $WORKDIR

bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 3 $CONFIG $WORKDIR

bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 4 $CONFIG $WORKDIR

bash ./scripts/ucit_llava_v15pf_lora/train_task.sh 5 $CONFIG $WORKDIR

