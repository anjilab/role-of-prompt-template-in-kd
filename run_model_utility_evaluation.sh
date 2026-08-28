BASE_PATH="/home/minillm_ai_safety"
LOG_FILE="evaluation_utility_model.log"
CONFIG_FILE="${BASE_PATH}/model_evaluation_configs.yaml"

get_config_value() {
    local model_key=$1
    local field=$2
    local default_value=${3-}
    python3 -c '
import yaml
import sys

config_file, model_key, field, default_value = sys.argv[1:5]

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

if model_key in config and "model" in config[model_key]:
    value = config[model_key]["model"].get(field, default_value)
    if value is None:
        value = default_value
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    print(value)
else:
    print(f"Error value: Fix your config file! No entry found for {model_key} or missing model field.", file=sys.stderr)
    sys.exit(1)
' "$CONFIG_FILE" "$model_key" "$field" "$default_value"
}

count_gpus() {
    local cuda_devices=$1
    if [[ -z "$cuda_devices" ]]; then
        echo ""
        return
    fi
    echo "$cuda_devices" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' '
}

if [ $# -eq 0 ]; then
    MODEL_CONFIGS=(
        # "llama_3_3b_it_kd_criscross_dolly_prompt_student"
        # "llama_3_3b_it_kd_criscross_llama_prompt_student"
        # "llama_3_8b_it"
        # "llama_3_8b_it_sft_dolly_prompt"
        # "llama_3_8b_it_sft_llama_prompt"
        # "gemma_2_9b_it"
        # "gemma_2_9b_it_sft_dolly_prompt"
        # "gemma_2_9b_it_sft_gemma_prompt"
        # "lama_3_3b_it_kd_curriculum_0_1_prompt_mix_student"
        # "lama_3_3b_it_kd_curriculum_0_4_prompt_mix_student"
        # "lama_3_3b_it_kd_curriculum_0_6_prompt_mix_student"
        # "lama_3_3b_it_kd_curriculum_0_2_prompt_mix_student"
        # "lama_3_3b_it_kd_curriculum_0_5_prompt_mix_student"
        # "lama_3_3b_it_kd_curriculum_0_8_prompt_mix_student"
        # "gemma_2_2b_it_curriculum_0_4_prompt_mix_student"
        # "gemma_2_2b_it_curriculum_0_5_prompt_mix_student"
        # "gemma_2_2b_it_curriculum_0_6_prompt_mix_student"
        # "gemma_2_2b_it_curriculum_0_8_prompt_mix_student"
        # "gemma2_2b_it_kd_criscross_dolly_prompt_student"
        # "gemma2_2b_it_kd_criscross_gemma_prompt_student"
        
        # "llama_3_8b_it_sft_llama_prompt_for_metamath"
        # "llama_3_3b_it_sft_llama_prompt_for_metamath"
        
        # "gemma_2_2b_it"
        # "gemma_2_9b_it"
        # "llama_3_8b_it_sft_llama_prompt_for_metamath"
        # "qwen_3_4b_it"
        # "qwen_3_2b_it"
        # "qwen_3_9b_it"
        # "llama_3_3b_it_sft_llama_prompt_for_metamath"
        # "llama_3_3b_it"
        # "llama_3_8b_it"
        # "gemma_2_2b_it"
        # "gemma_2_9b_it"
        # "llama_3_8b_it_sft_llama_prompt_for_metamath"
        # "qwen_2_7b_it"
        # "qwen_2_3b_it"
        
        # "qwen_2_3b_it"
        # "qwen_2_7b_it"
        # "qwen_2_3b_it_sft_metamath_qwen_prompt"
        # "qwen_2_7b_it_sft_metamath_qwen_prompt"

        # "gemma_2_2b_it"
        # "gemma_2_9b_it"
        # "llama_3_3b_it"
        # "llama_3_8b_it"
        # "llama_3_3b_it_sft_llama_prompt_for_metamath"
        # "llama_3_8b_it_sft_llama_prompt_for_metamath"
        # "qwen_2_3b_it_sft_metamath_qwen_prompt"
        # "qwen_2_7b_it_sft_metamath_qwen_prompt"
        # "llama_3_3b_it_sft_llama_prompt_for_metamath"
        # "llama_3_3b_it"
        # "llama_3_3b_it_sft_llama_prompt_for_medqa"
        # "llama_3_3b_it_sft_llama_prompt_for_medqa_ckpt2"
        # "llama_3_3b_it_sft_llama_prompt_for_medqa_ckpt3"

    
        # "llama_3_8b_it_sft_llama_prompt_for_medqa"
        # "llama_3_8b_it_sft_llama_prompt_for_medqa_ckpt2"
        # "llama_3_8b_it_sft_llama_prompt_for_medqa_ckpt3"
       
        # "llama_3_8b_it"

        # BELOW 3 FOR 8b MODELS MEDQA PROMPT IS DONE DEAL, NO NEED TO EVALUATE AND TRAIN AGAIN.
        # "llama_3_8b_it_sft_medqa_prompt_for_medqa"
        # "llama_3_8b_it_sft_medqa_prompt_for_medqa_ckpt2"
        # "llama_3_8b_it_sft_medqa_prompt_for_medqa_ckpt3"

        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa"
        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt2"
        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt3"
        # "llama_3_8b_it_kd_teacher_llama_prompt_student_llama_for_medqa"
        # "llama_3_8b_it_kd_teacher_llama_prompt_student_llama_for_medqa_ckpt2"
        # "llama_3_8b_it_kd_teacher_llama_prompt_student_llama_for_medqa_ckpt3"

        #### All evaluation is done with llama prompt.  ##### 
        # This is for 3B sft models with medqa prompt
        # "llama_3_3b_it_sft_medqa_prompt_for_medqa"
        # "llama_3_3b_it_sft_medqa_prompt_for_medqa_ckpt2"
        # "llama_3_3b_it_sft_medqa_prompt_for_medqa_ckpt3"
        # This is for 3B sft models with llama prompt
        # "llama_3_3b_it_sft_llama_prompt_for_medqa"
        # "llama_3_3b_it_sft_llama_prompt_for_medqa_ckpt2"
        # "llama_3_3b_it_sft_llama_prompt_for_medqa_ckpt3"
        ##### This is for kd models 
        # Traind with medqa prompt in medqa dataset.
        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa"
        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt2"
        # "llama_3_3b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt3"
        # Traine with llama prompt in chat dataset.
        # "llama_3_3b_it_kd_teacher_llama_prompt_student_llama_for_medqa"
        # "llama_3_3b_it_kd_teacher_llama_prompt_student_llama_for_medqa_ckpt2"
        # "llama_3_3b_it_kd_teacher_llama_prompt_student_llama_for_medqa_ckpt3"
        

        #### All evaluation is done with llama prompt.  ##### 
        ### For gemma base models
        # "gemma_2_2b_it"
        # "gemma_2_9b_it"

        # "gemma_2_2b_it_kd_teacher_gemma_prompt_student_gemma_for_medqa"
        # "gemma_2_2b_it_kd_teacher_gemma_prompt_student_gemma_for_medqa_ckpt2"
        # "gemma_2_2b_it_kd_teacher_gemma_prompt_student_gemma_for_medqa_ckpt3"
        # "gemma_2_2b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa"
        # "gemma_2_2b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt2"
        # "gemma_2_2b_it_kd_teacher_medqa_prompt_student_medqa_prompt_for_medqa_ckpt3"


        # # This is for 2B sft models with gemma prompt
        # "gemma_2_2b_it_sft_gemma_prompt_for_medqa"
        # "gemma_2_2b_it_sft_gemma_prompt_for_medqa_ckpt2"
        # "gemma_2_2b_it_sft_gemma_prompt_for_medqa_ckpt3"
        # # This is for 2B sft models with medqa prompt
        # "gemma_2_2b_it_sft_medqa_prompt_for_medqa"
        # "gemma_2_2b_it_sft_medqa_prompt_for_medqa_ckpt2"
        # "gemma_2_2b_it_sft_medqa_prompt_for_medqa_ckpt3"

        # "gemma_2_9b_it_sft_gemma_prompt_for_medqa"
        # "gemma_2_9b_it_sft_gemma_prompt_for_medqa_ckpt2"
        # "gemma_2_9b_it_sft_gemma_prompt_for_medqa_ckpt3"
        # "gemma_2_9b_it_sft_medqa_prompt_for_medqa"
        # "gemma_2_9b_it_sft_medqa_prompt_for_medqa_ckpt2"
        # "gemma_2_9b_it_sft_medqa_prompt_for_medqa_ckpt3"

        # "qwen_2_7b_it"
        # "qwen_2_3b_it"

        # "qwen_2_7b_it_sft_qwen_prompt_for_dolly"
        # "qwen_2_7b_it_sft_dolly_prompt_for_dolly"
        # "qwen_2_3b_it_sft_qwen_prompt_for_dolly"
        # "qwen_2_3b_it_sft_dolly_prompt_for_dolly"

        # "qwen_2_3b_it_kd_qwen_prompt_for_dolly"
        # "qwen_2_3b_it_kd_dolly_prompt_for_dolly"

        # For medqa
        # "qwen_2_7b_it"
        # "qwen_2_3b_it"

        # "qwen_2_3b_it_sft_qwen_prompt_for_medqa_ckpt1"
        # "qwen_2_3b_it_sft_qwen_prompt_for_medqa_ckpt2"
        # "qwen_2_3b_it_sft_qwen_prompt_for_medqa_ckpt3"

        # "qwen_2_3b_it_sft_medqa_prompt_for_medqa_ckpt1"
        # "qwen_2_3b_it_sft_medqa_prompt_for_medqa_ckpt2"
        # "qwen_2_3b_it_sft_medqa_prompt_for_medqa_ckpt3"


        # "qwen_2_7b_it_sft_qwen_prompt_for_medqa_ckpt1"
        # "qwen_2_7b_it_sft_qwen_prompt_for_medqa_ckpt2"
        # "qwen_2_7b_it_sft_qwen_prompt_for_medqa_ckpt3"

        # "qwen_2_7b_it_sft_medqa_prompt_for_medqa_ckpt1"
        # "qwen_2_7b_it_sft_medqa_prompt_for_medqa_ckpt2"

        # "qwen_2_3b_it_kd_medqa_prompt_for_medqa_ckpt1"
        # "qwen_2_3b_it_kd_medqa_prompt_for_medqa_ckpt2"
        # "qwen_2_3b_it_kd_medqa_prompt_for_medqa_ckpt3"

        # "qwen_2_3b_it_kd_qwen_prompt_for_medqa_ckpt1"
        # "qwen_2_3b_it_kd_qwen_prompt_for_medqa_ckpt2"
        # "qwen_2_3b_it_kd_qwen_prompt_for_medqa_ckpt3"

        # "qwen_2_3b_it_rkl_dolly_prompt_for_dolly"
        "qwen_2_3b_it_rkl_qwen_prompt_for_dolly"
    )
else
    MODEL_CONFIGS=("$@")
fi
echo $MODEL_CONFIGS



echo "Starting Evaluation ......"
for config_key in "${MODEL_CONFIGS[@]}"; do
    echo ""
    echo "Reading configuration for: ${config_key}"
    
    CKPT_NAME=$(get_config_value "$config_key" "ckpt_name")
    CKPT_PATH=$(get_config_value "$config_key" "ckpt_path")
    PEFT_CKPT_NAME=$(get_config_value "$config_key" "peft_ckpt_name")
    PEFT_CKPT=$(get_config_value "$config_key" "peft_ckpt_path")
    CUDA_DEVICES=$(get_config_value "$config_key" "CUDA_DEVICES")
    EVAL_SCRIPT=$(get_config_value "$config_key" "eval_script")
    GPUS_PER_NODE=$(get_config_value "$config_key" "gpus_per_node" "")
    SAVE_ROOT=$(get_config_value "$config_key" "save_root" "")
    EVAL_METHOD=$(get_config_value "$config_key" "eval_method" "")
    EVAL_FAMILY=$(get_config_value "$config_key" "eval_family" "")
    EVAL_TECHNIQUE=$(get_config_value "$config_key" "eval_technique" "")
    PROMPT_TYPE=$(get_config_value "$config_key" "prompt_type" "")
    MASTER_PORT=$(get_config_value "$config_key" "master_port" "")
    if [[ -z "$GPUS_PER_NODE" ]]; then
        GPUS_PER_NODE=$(count_gpus "$CUDA_DEVICES")
    fi
    
    # Check if config was found
    if [[ -z "$CKPT_NAME" || -z "$CKPT_PATH" ]]; then
        echo "Error: Configuration not found for ${config_key}"
        exit 1
    fi
    


    {
        echo "========================================"
        echo "Config Key  : $config_key"
        echo "Base Model  : $CKPT_NAME"
        echo "Base Path   : $CKPT_PATH"
        echo "PEFT Name   : $PEFT_CKPT_NAME"
        echo "PEFT Path   : $PEFT_CKPT"
        echo "CUDA Devices: $CUDA_DEVICES"
        echo "GPUs/Node   : $GPUS_PER_NODE"
        echo "Save Root   : $SAVE_ROOT"
        echo "Eval Method : $EVAL_METHOD"
        echo "Eval Family : $EVAL_FAMILY"
        echo "Technique   : $EVAL_TECHNIQUE"
        echo "Prompt Type : $PROMPT_TYPE"
        echo "Eval Script : $EVAL_SCRIPT"
        echo "========================================"
    } | tee -a "$LOG_FILE"


    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    EVAL_GPUS_PER_NODE="$GPUS_PER_NODE" \
    EVAL_SAVE_ROOT="$SAVE_ROOT" \
    EVAL_METHOD="$EVAL_METHOD" \
    EVAL_FAMILY="$EVAL_FAMILY" \
    EVAL_TECHNIQUE="$EVAL_TECHNIQUE" \
    EVAL_PROMPT_TYPE="$PROMPT_TYPE" \
    MASTER_PORT_OVERRIDE="$MASTER_PORT" \
    bash "$EVAL_SCRIPT" "$PEFT_CKPT_NAME" "$PEFT_CKPT" "$CKPT_NAME" "$CKPT_PATH" "$CUDA_DEVICES" "$GPUS_PER_NODE" "$SAVE_ROOT" "$EVAL_METHOD" "$EVAL_FAMILY" "$EVAL_TECHNIQUE" "$PROMPT_TYPE"

done

echo ""
echo "Evaluation completed. Check ${LOG_FILE} for details."

# bash /home/minillm_ai_safety/run_medqa.sh
