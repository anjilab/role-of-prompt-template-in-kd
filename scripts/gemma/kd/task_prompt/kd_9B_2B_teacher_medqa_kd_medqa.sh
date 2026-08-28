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


CKPT_NAME="gemma-2-2b-it"
CKPT="/media/drive2/models/gemma-models/gemma-2-2b-it"

TEACHER_CKPT_NAME="gemma-2-9b-it"
TEACHER_CKPT="/media/drive2/models/gemma-models/gemma-2-9b-it"

TEACHER_PEFT_CKPT_NAME="sft_gemma9b-it_med-qa_medqa-prompt_e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05_20260513_040514_312"
TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/gemma2/train/sft/gemma9b-it/med-qa/medqa-prompt/e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05/20260513_040514/312"
# data
DATA_DIR="${BASE_PATH}/processed_data/medqa/full/gemma2"
DATA_TYPE="med-qa"
# hp

BATCH_SIZE=1             
GRAD_ACC=32          
EVAL_BATCH_SIZE=1
LR=1e-4                 

TYPE="kd"
KD_RATIO=0.5 

EPOCH=3
MAX_PROMPT_LENGTH=512
MAX_LENGTH=1024

# runtime
SAVE_PATH="${DRIVE_PATH}/results/gemma2/train/kd/chat/9-2B/gemma-prompt/${DATA_TYPE}/${KD_RATIO}"
# seed
SEED=0
SEED_ORDER=0
WANDB_RUN_NAME="kd_2B_9B_teacher_medqa_lora_medqa_prompt_medqa"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gemma2"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-type ${DATA_TYPE}"
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
# OPTS+=" --peft-lora-r 16"
# OPTS+=" --peft-lora-alpha 32"
OPTS+=" --peft-lora-dropout 0.05"

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