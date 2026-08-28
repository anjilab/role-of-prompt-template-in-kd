#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT_OVERRIDE:-20128}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${6:-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="/home/minillm_ai_safety"
DRIVE_PATH="/media/drive2/distillation/safety-utility-pg"

CKPT_NAME=${4-"llama-3B"}
CKPT=${5-"/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"}
PEFT_CKPT_NAME=${2}
PEFT_CKPT=${3}
# data
DATA_NAMES="vicuna"
DATA_DIR="${BASE_PATH}/processed_data/llama3/vicuna/full/llama3-chat"

# hp
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE_OVERRIDE:-2}
# runtime
# SAVE_PATH="${DRIVE_PATH}/results/llama3/eval_main/kd/8-2B/llama-prompt"
# SAVE_PATH="${DRIVE_PATH}/results/llama3/eval_main/sft/3B/llama-prompt"
# SAVE_PATH="${DRIVE_PATH}/results/llama3/eval_main/sft/8B/llama-prompt"
# SAVE_PATH="${DRIVE_PATH}/results/llama3/eval_main/kd/8-2B/mixing/llama-prompt"
SAVE_PATH=${SAVE_PATH_OVERRIDE:-"${DRIVE_PATH}/results/llama3/eval_main/kd/8-2B/criss-cross/llama-prompt"}

TYPE="eval_main"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama3"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 1024"
OPTS+=" --max-prompt-length 512"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${1-0}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_bf16.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.7"
OPTS+=" --temperature 1.0"

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 8"
OPTS+=" --peft-lora-alpha 16"
if [[ "${PEFT_CKPT_NAME}" != "test" ]]; then
    OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
    OPTS+=" --peft-path ${PEFT_CKPT}"
fi


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
