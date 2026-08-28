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

CKPT_NAME="llama-3B"
CKPT="/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
# CKPT="/media/drive2/models/llama-models/Llama-3.2-3B"

TEACHER_CKPT_NAME="llama-8B"
TEACHER_CKPT="/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"

# TEACHER_PEFT_CKPT_NAME="sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068"
# TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama3/train/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068"

TEACHER_PEFT_CKPT_NAME="sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068"
TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama3/train/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068"
# MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama3"
# hp

BATCH_SIZE=1             
GRAD_ACC=16          
EVAL_BATCH_SIZE=2
LR=2e-5                  

TYPE="kd"
# TYPE="rkl" # FOr rkl, feb 9 expt
# TYPE="fkl+rkl" # For fkl+rkl, feb 18 expt
# TYPE="jsd" # For jsd, feb 25 expt
WANDB_RUN_NAME="kd_3B_8B_teacher_dolly_lora_llama_fkl_teacher_criscross_evaluation"
KD_RATIO=0.5 
# KD_RATIO=1.0 # FOR RKL, WITH 1.0, FEB 11 expt

EPOCH=8
MAX_PROMPT_LENGTH=512
MAX_LENGTH=1024

# runtime
# SAVE_PATH="${DRIVE_PATH}/results/llama3/train/kd/without_it_student_models/8-3B/dolly-prompt/${TYPE}/${KD_RATIO}"
SAVE_PATH="${DRIVE_PATH}/results/llama3/train/kd/chat/8-3B/dolly-prompt/teacher-criscross-evaluation/${TYPE}/${KD_RATIO}"

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
OPTS+=" --model-type llama3"
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
OPTS+=" --top-k 0" # Weighted sampling. 
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