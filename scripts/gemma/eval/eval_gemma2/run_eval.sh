base_path="/home/minillm_ai_safety"
PEFT_CKPT_NAME=$1
PEFT_CKPT_PATH=$2
CKPT_NAME=$3
CKPT_PATH=$4
CUDA_DEVICES=${5:-${CUDA_VISIBLE_DEVICES}}
GPUS_PER_NODE=${6:-${EVAL_GPUS_PER_NODE:-4}}
SAVE_ROOT=${7:-${EVAL_SAVE_ROOT:-"/media/drive2/distillation/safety-utility-pg/results/gemma2/eval_main"}}
EVAL_METHOD=${8:-${EVAL_METHOD:-}} # sft, kd, base
EVAL_FAMILY=${9:-${EVAL_FAMILY:-2b}} # 3b, 2b, 8b, 9b, 8-3b, 9-2b
EVAL_TECHNIQUE=${10:-${EVAL_TECHNIQUE:-}} # empty or criss-cross or curriculum 
PROMPT_TYPE=${11:-${EVAL_PROMPT_TYPE:-}} # llama-prompt, dolly-prompt

get_eval_batch_size() {
    local family
    family=$(echo "$EVAL_FAMILY" | tr '[:upper:]' '[:lower:]')

    case "$family" in
        8b)
            echo 2
            ;;
        9b)
            echo 1
            ;;
        3b|2b|8-3b|9-2b)
            echo 4
            ;;
        *)
            echo 2
            ;;
    esac
}

build_save_path() {
    # Issue with this saves "//" for base path.
    #  To save the path based on save_root provided from mode_evaluation_configs.yaml and the eval method, family, technique and prompt type. 
    #  The path will be in the format of save_root/eval_method/eval_family/eval_technique/prompt_type
    local save_path="${SAVE_ROOT}/${EVAL_METHOD}/${EVAL_FAMILY}"

    if [[ -n "$EVAL_TECHNIQUE" ]]; then
        save_path="${save_path}/${EVAL_TECHNIQUE}"
    fi

    save_path="${save_path}/${PROMPT_TYPE}"

    echo "$save_path"
}

# build_save_path() {
#     # Build save_root/eval_method/eval_family/eval_technique/prompt_type,
#     # skipping optional empty components such as eval_family for base models.
#     local save_path="${SAVE_ROOT%/}"
#     local path_part

#     for path_part in "$EVAL_METHOD" "$EVAL_FAMILY" "$EVAL_TECHNIQUE" "$PROMPT_TYPE"; do
#         if [[ -n "$path_part" ]]; then
#             save_path="${save_path}/${path_part}"
#         fi
#     done

#     echo "$save_path"
# }


if [ "$PEFT_CKPT_NAME" == "gemma-2-2b-it" -o "$PEFT_CKPT_NAME" == "gemma-2-9b-it" ]
then
    PEFT_CKPT_NAME="test"
    PEFT_CKPT_PATH="test"
fi

echo "Evaluating PEFT_CKPT_NAME: $PEFT_CKPT_NAME"
echo "EVAL_FAMILY: $EVAL_FAMILY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "Save Path: $(build_save_path)"

# for data in dolly self_inst vicuna sinst metamath
# for data in dolly self_inst vicuna sinst
# for data in self_inst
# for data in metamath 
for data in medqa 

do
    for seed in 10
    do
        EVAL_BATCH_SIZE=$(get_eval_batch_size "$data")
        SAVE_PATH=$(build_save_path)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        EVAL_BATCH_SIZE_OVERRIDE="$EVAL_BATCH_SIZE" \
        SAVE_PATH_OVERRIDE="$SAVE_PATH" \
        bash "${base_path}/scripts/gemma/eval/eval_gemma2/eval_main_${data}.sh" "$seed" "$PEFT_CKPT_NAME" "$PEFT_CKPT_PATH" "$CKPT_NAME" "$CKPT_PATH" "$GPUS_PER_NODE"
    done
done
