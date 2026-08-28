#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=20127
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="/home/minillm_ai_safety"
DRIVE_PATH="/media/drive2/distillation/safety-utility-pg"

CKPT_NAME="qwen2.5-3B-it"
CKPT="/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct"

TEACHER_CKPT_NAME="qwen2.5-7B-it"
TEACHER_CKPT="/media/drive2/models/qwen-models/Qwen2.5-7B-Instruct"

TEACHER_PEFT_CKPT_NAME="sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246"
TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/qwen2/train/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246"

# MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen2"
DATA_TYPE="dolly"
# hp

BATCH_SIZE=1             
GRAD_ACC=16          
EVAL_BATCH_SIZE=2
LR=2e-5                  

TYPE="rkl"
# TYPE="rkl" # FOr rkl, feb 9 expt
# TYPE="fkl+rkl" # For fkl+rkl, feb 18 expt
# TYPE="jsd" # For jsd, feb 25 expt
WANDB_RUN_NAME="rkl_3B_7B_teacher_dolly_student_dolly_for_dolly_dataset"
KD_RATIO=0.5 
# KD_RATIO=1.0 # FOR RKL, WITH 1.0, FEB 11 expt

EPOCH=8
MAX_PROMPT_LENGTH=512
MAX_LENGTH=1024

# runtime
SAVE_PATH="${DRIVE_PATH}/results/qwen2/train/${TYPE}/chat/7-3B/dolly/dolly-prompt/${KD_RATIO}"

# seed
SEED=0


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen2"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 200"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCH}"
OPTS+=" --kd-ratio ${KD_RATIO}"
OPTS+=" --wandb-run-name ${WANDB_RUN_NAME}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 8"
OPTS+=" --peft-lora-alpha 16"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_bf16.json"
# type
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune_bits_and_bytes.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}