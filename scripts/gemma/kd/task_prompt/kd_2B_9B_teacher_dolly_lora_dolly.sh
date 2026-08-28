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

# TEACHER_PEFT_CKPT_NAME="sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712"
# TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/gemma2/train/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712"

# Student = Task/Non-chat prompt and Teacher prompt = Gemma prompt (chat prompt)
TEACHER_PEFT_CKPT_NAME="sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424"
TEACHER_PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/gemma2/train/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424"

# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gemma2"
# hp
BATCH_SIZE=1              
GRAD_ACC=16            
EVAL_BATCH_SIZE=4
LR=2e-5                

# TYPE="rkl" # For rkl, feb 10 expt
TYPE="kd" # For kd and ration = 1.0, feb 14 expt
# TYPE="fkl+rkl" # For fkl+rkl, feb 18 expt
# WANDB_RUN_NAME="gemma_kd_3B_8B_teacher_dolly_lora_dolly_fkl_basemodels"
WANDB_RUN_NAME="gemma_kd_3B_8B_teacher_gemma_lora_student_dolly_teacher_criscross_evaluation"

# WANDB_RUN_NAME="gemma_kd_2B_9B_teacher_dolly_lora_dolly_kd_ratio_1_0"

# KD_RATIO=1.0 # For KD, with 1.0, feb 14 expt
KD_RATIO=0.5
EPOCH=8
MAX_PROMPT_LENGTH=512
MAX_LENGTH=1024


# runtime
# SAVE_PATH="${DRIVE_PATH}/results/gemma2/train/kd/chat/9-2B/dolly-prompt/${TYPE}/${KD_RATIO}"
SAVE_PATH="${DRIVE_PATH}/results/gemma2/train/kd/without_it_student_models/9-2B/dolly-prompt/teacher-criscross-evaluation/${TYPE}/${KD_RATIO}"


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
OPTS+=" --model-type gemma2"
OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
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
