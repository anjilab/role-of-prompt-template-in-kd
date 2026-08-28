BASE_PATH="/home/minillm_ai_safety"
BASE_PATH_HARMBENCH="${BASE_PATH}/HarmBench"
base_dir="${BASE_PATH}/data/harm_bench"

max_new_tokens=512
incremental_update="False"
test_cases_path="$base_dir/test_cases/test_cases.json"
behaviors_path="$BASE_PATH_HARMBENCH/data/behavior_datasets/harmbench_behaviors_text_test.csv"

# Declare associative arrays
declare -A ckpt_names      
declare -A output_dirs         
    

add_model() {
    local ckpt_name="$1"        
    local output_dir="$2"               
    
    ckpt_names["$ckpt_name"]="$ckpt_name"
    output_dirs["$ckpt_name"]="$output_dir"
}


# add_model "gemma_2_2b_it"  "gemma2/chat/base/2B" # 12.4
# add_model "gemma_2_9b_it" "gemma2/chat/base/9B" # 14.3
# add_model "gemma_2_2b_it_sft_dolly_prompt" "gemma2/chat/sft/doll_prompt/2B" # 16.9

# add_model "gemma_2_2b_it" "gemma2/chat/base-gemma2B"
# add_model "gemma_2_9b_it" "gemma2/chat/base-gemma9B"
# add_model "gemma_2_2b_it_sft_gemma_prompt" "gemma2/chat/2B/gemma_prompt/sft-gemma2B"
# add_model "gemma_2_2b_it_sft_dolly_prompt" "gemma2/chat/2B/dolly_prompt/sft-gemma2B"
# add_model "gemma_2_9b_it_sft_gemma_prompt" "gemma2/chat/9B/gemma_prompt/sft-gemma9B"
# add_model "gemma_2_9b_it_sft_dolly_prompt" "gemma2/chat/9B/dolly_prompt/sft-gemma9B"
# add_model "gemma_2_9b_it_kd_gemma_teacher_gemma" "gemma2/chat/9_2B/kd_gemma_prompt_teacher_gemma/kd-gemma-prompt-teacher-gemma"
# add_model "gemma_2_9b_it_kd_dolly_teacher_dolly" "gemma2/chat/9_2B/kd_dolly_prompt_teacher_dolly/kd-dolly-prompt-teacher-dolly"

# add_model "gemma_2_2b_it"  "gemma2/chat/base/2B" # 12.4
# add_model "gemma_2_2b_it_sft_dolly_prompt" "gemma2/chat/sft/dolly_prompt/2B" # 16.9
# add_model "gemma_2_2b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "gemma2/chat/kd/0.5/dolly_prompt/2B/teacher-dolly-prompt" # 16.9
# add_model "gemma_2_2b_it_kd_0.5_gemma_prompt_teacher_gemma_prompt" "gemma2/chat/kd/0.5/gemma_prompt/2B/teacher-gemma-prompt" # 16.9

# add_model "gemma_2_9b_it" "gemma2/chat/base/9B" # 14.3

# add_model "llama_3_3b_it"   "llama3/chat/base/3B"
# add_model "llama_3_8b_it"   "llama3/chat/base/8B"



# add_model "llama_3_3b_it"   "llama3/chat/base/3B"
# add_model "llama_3_8b_it"   "llama3/chat/base/8B"
# add_model "llama_3_8b_it_sft_llama_prompt" "llama3/chat/8B/llama_prompt/sft-llama8B"
# add_model "llama_3_8b_it_sft_dolly_prompt" "llama3/chat/8B/dolly_prompt/sft-llama8B"

# add_model "llama_3_3b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "llama3/chat/8_3B/kd_dolly_prompt_teacher_dolly/kd-dollyy-prompt-teacher-dollyy"
# add_model "llama_3_3b_it_kd_0.5_llama_prompt_teacher_llama_prompt" "llama3/chat/8_3B/kd_llama_prompt_teacher_llama/kd-llama-prompt-teacher-llama"

# add_model "llama_3_3b_it_sft_dolly_prompt" "llama3/chat/3B/dolly_prompt/sft-llama3B" 
# add_model "llama_3_3b_it_sft_llama_prompt" "llama3/chat/3B/llama_prompt/sft-llama3B" 


# add_model "gemma_2_2b_it_sft_dolly_prompt" "gemma2/chat/sft/dolly_prompt/2B" # 16.9
# add_model "gemma_2_2b_it_sft_gemma_prompt" "gemma2/chat/sft/gemma_prompt/2B" # 16.9
# add_model "gemma_2_2b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "gemma2/chat/kd/0.5/dolly_prompt/2B/teacher-dolly-prompt" # 16.9
# add_model "gemma_2_2b_it_kd_0.5_gemma_prompt_teacher_gemma_prompt" "gemma2/chat/kd/0.5/gemma_prompt/2B/teacher-gemma-prompt" # 16.9

# add_model "gemma_2_9b_it" "gemma2/chat/base/9B" # 14.3

# add_model "llama_3_3b_it"   "llama3/chat/base/3B"
# add_model "llama_3_8b_it"   "llama3/chat/base/8B"


### DEC 30 RUNNING MODELS SEED 

# add_model "llama_3_3b_it" "llama3/chat/seed/base/3B"
# add_model "llama_3_3b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "llama3/chat/seed/8_3B/kd_dolly_prompt_teacher_dolly/kd-dolly-prompt-teacher-dolly"
# add_model "llama_3_3b_it_kd_0.5_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/8_3B/kd_llama_prompt_teacher_llama/kd-llama-prompt-teacher-llama"
# add_model "gemma_2_2b_it" "gemma2/chat/seed/base/2B"
# add_model "gemma_2_2b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "gemma2/chat/seed/kd/0.5/dolly_prompt/2B/teacher-dolly-prompt"
# add_model "gemma_2_2b_it_kd_0.5_gemma_prompt_teacher_gemma_prompt" "gemma2/chat/seed/kd/0.5/gemma_prompt/2B/teacher-gemma-prompt"


#### Jan 3 for sft teacher models
# add_model "llama_3_8b_it_sft_llama_prompt" "llama3/chat/sft/sft_llama_prompt_8b-it"
# add_model "llama_3_8b_it_sft_dolly_prompt" "llama3/chat/sft/sft_dolly_prompt_8b-it"
# add_model "gemma_2_9b_it_sft_gemma_prompt" "gemma2/chat/sft/sft_gemma_prompt_9b-it"
# add_model "gemma_2_9b_it_sft_dolly_prompt" "gemma2/chat/sft/sft_dolly_prompt_9b-it"


# add_model "llama_3_3b_it_kd_0.5_dolly_prompt_teacher_dolly_prompt" "llama3/chat/seed/revised/8_3B/kd_dolly_prompt_teacher_dolly/kd-dolly-prompt-teacher-dolly"
# add_model "llama_3_3b_it_kd_0.5_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/revised/8_3B/kd_llama_prompt_teacher_llama/kd-llama-prompt-teacher-llama"
# add_model "llama_3_3b_it_rkl_0.5_dolly_prompt_teacher_llama_prompt" "llama3/chat/seed/revised/8_3B/rkl_dolly_prompt_teacher_llama/kd-dolly-prompt-teacher-llama"
# add_model "llama_3_3b_it_rkl_0.5_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/revised/8_3B/rkl_llama_prompt_teacher_llama/kd-llama-prompt-teacher-llama"
# add_model "gemma_2_2b_it_rkl_0.5_dolly_prompt_teacher_dolly_prompt" "gemma2/chat/seed/revised/rkl/0.5/dolly_prompt/2B/teacher-dolly-prompt"
# add_model "gemma_2_2b_it_rkl_0.5_gemma_prompt_teacher_gemma_prompt" "gemma2/chat/seed/revised/rkl/0.5/gemma_prompt/2B/teacher-gemma-prompt"

# add_model "llama_3_3b_it_fkl_rkl_0.5_dolly_prompt_teacher_llama_prompt" "llama3/chat/seed/revised/8_3B/fkl+rkl_dolly_prompt_teacher_llama/kd-dolly-prompt-teacher-llama"
# add_model "llama_3_3b_it_fkl_rkl_0.5_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/revised/8_3B/fkl+rkl_llama_prompt_teacher_llama/kd-llama-prompt-teacher-llama"


# add_model "llama_3_3b_it_sft_dolly_prompt" "llama3/chat/sft/sft_dolly_prompt_3b-it"
# add_model "llama_3_3b_it_sft_llama_prompt" "llama3/chat/sft/sft_llama_prompt_3b-it"
# add_model "llama_3_3b_it_sft_dolly_prompt" "llama3/chat/sft/sft_dolly_prompt_3b-it"
# add_model "gemma_2_2b_it_sft_dolly_prompt" "gemma2/chat/sft/v2/sft_dolly_prompt_2b-it"
# add_model "gemma_2_2b_it_sft_gemma_prompt" "gemma2/chat/sft/v2/sft_gemma_prompt_2b-it"


# add_model "llama_3_3b_it_mix_0_2_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/mixing/8_3B/mix-0_2-llama-prompt-teacher-llama"
# add_model "llama_3_3b_it_mix_0_5_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/mixing/8_3B/mix-0_5-llama-prompt-teacher-llama"
# add_model "llama_3_3b_it_mix_0_8_llama_prompt_teacher_llama_prompt" "llama3/chat/seed/mixing/8_3B/mix-0_8-llama-prompt-teacher-llama"



# add_model "llama_3_3b_it_crisscross_dolly_prompt_teacher_llama_prompt" "llama3/chat/seed/crisscross/8_3B/crisscross-dolly-prompt-teacher-llama"
# add_model "llama_3_3b_it_crisscross_llama_prompt_teacher_dolly_prompt" "llama3/chat/seed/crisscross/8_3B/crisscross-llama-prompt-teacher-dolly"

# add_model "llama_3_8b_it" "llama3/chat/base/8B-it-again"
# add_model "llama_3_8b_it_sft_dolly_prompt" "llama3/chat/sft/v2/sft_dolly_prompt_8b-it-again"
# add_model "llama_3_8b_it_sft_llama_prompt" "llama3/chat/sft/v2/sft_llama_prompt_8b-it-again"


# add_model "llama_3_3b_it_curriculum_0_1_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_1"
# add_model "llama_3_3b_it_curriculum_0_2_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_2"
# add_model "llama_3_3b_it_curriculum_0_4_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_4"
# add_model "llama_3_3b_it_curriculum_0_5_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_5"
# add_model "llama_3_3b_it_curriculum_0_6_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_6"
# add_model "llama_3_3b_it_curriculum_0_8_prompt_mixing_student" "llama3/chat/seed/mixing/8_3B/curriculum_mix_0_8"


# add_model "gemma_2_2b_it_crisscross_teacher_gemma_student_dolly" "gemma2/chat/9_2B/teacher-criscross-evaluation/teacher_gemma_student_dolly"
# add_model "gemma_2_2b_it_crisscross_teacher_dolly_student_gemma" "gemma2/chat/9_2B/teacher-criscross-evaluation/teacher_dolly_student_gemma"

# add_model "gemma_2_2b_it_curriculum_0_8_prompt_mixing_student" "gemma2/chat/9_2B/gemma-prompt/mixing/kd/0.5/curriculum_0.8" 
# add_model "gemma_2_2b_it_curriculum_0_6_prompt_mixing_student" "gemma2/chat/9_2B/gemma-prompt/mixing/kd/0.5/curriculum_0.6"
# add_model "gemma_2_2b_it_curriculum_0_5_prompt_mixing_student" "gemma2/chat/9_2B/gemma-prompt/mixing/kd/0.5/curriculum_0.5"
# add_model "gemma_2_2b_it_curriculum_0_4_prompt_mixing_student" "gemma2/chat/9_2B/gemma-prompt/mixing/kd/0.5/curriculum_0.4"

# add_model "qwen_2b_chat" "qwen3/chat/base/2B"
# add_model "qwen_9b_chat" "qwen3/chat/base/9B"
# add_model "qwen_4b_chat" "qwen3/chat/base/4B"


# add_model "qwen_2_5_3b_it" "qwen2/chat/base/3B"
# add_model "qwen_2_5_7b_it" "qwen2/chat/base/7B"
# add_model "llama_3_1b_it" "llama3/chat/base/1B"
# add_model "llama_3_3b_it_sft_medqa_prompt_for_medqa" "llama3/chat/sft/medqa/medqa_prompt/3B/sft-medqa-prompt-3B-it-r=8-16"
# add_model "llama_3_3b_it_sft_llama_prompt_for_medqa" "llama3/chat/sft/medqa/llama_prompt/3B/sft-llama-prompt-3B-it-r=8-16"
# add_model "llama_3_3b_it_kd_0.5_medqa_prompt_teacher_medqa_prompt_for_medqa" "llama3/chat/kd/0.5/medqa/medqa_prompt/3B/teacher-medqa-prompt/kd_0.5_medqa_prompt_teacher_medqa_prompt_for_medqa"
# add_model "llama_3_3b_it_kd_0.5_llama_prompt_teacher_llama_prompt_for_medqa" "llama3/chat/kd/0.5/medqa/llama_prompt/3B/teacher-llama-prompt/kd_0.5_llama_prompt_teacher_llama_prompt_for_medqa"


# add_model "gemma_2_2b_it_kd_0.5_medqa_prompt_teacher_medqa_prompt_for_medqa" "gemma2/chat/kd/0.5/medqa/medqa_prompt/2B/teacher-medqa-prompt/kd_0.5_medqa_prompt_teacher_medqa_prompt_for_medqa"
# add_model "gemma_2_2b_it_kd_0.5_gemma_prompt_teacher_gemma_prompt_for_medqa" "gemma2/chat/kd/0.5/medqa/gemma_prompt/2B/teacher-gemma-prompt/kd_0.5_gemma_prompt_teacher_gemma_prompt_for_medqa"

# add_model "qwen_2_5_3b_it" "qwen2/chat/base/3B"
# add_model "qwen_2_5_7b_it" "qwen2/chat/base/7B"
# add_model "qwen_2_5_1_5b_it" "qwen2/chat/base/1.5B"

# add_model "qwen_2_5_3b_it_sft_qwen_prompt" "qwen2/chat/sft/dolly/qwen_prompt/3B"
# add_model "qwen_2_5_3b_it_sft_dolly_prompt" "qwen2/chat/sft/dolly/dolly_prompt/3B"
# add_model "qwen_2_5_7b_it_sft_qwen_prompt" "qwen2/chat/sft/dolly/qwen_prompt/7B"
# add_model "qwen_2_5_7b_it_sft_dolly_prompt" "qwen2/chat/sft/dolly/dolly_prompt/7B"

# add_model "qwen_2_5_3b_it_kd_qwen_prompt" "qwen2/chat/kd/dolly/qwen_prompt/3B"
# add_model "qwen_2_5_3b_it_kd_dolly_prompt" "qwen2/chat/kd/dolly/dolly_prompt/3B"


# add_model "qwen_2_5_3b_it_sft_qwen_prompt_medqa"  "qwen2/chat/sft/medqa/qwen_prompt/3B"
# add_model "qwen_2_5_3b_it_sft_medqa_prompt_medqa"  "qwen2/chat/sft/medqa/medqa_prompt/3B"
# add_model "qwen_2_5_7b_it_sft_qwen_prompt_medqa" "qwen2/chat/sft/medqa/qwen_prompt/7B"
# add_model "qwen_2_5_7b_it_sft_medqa_prompt_medqa" "qwen2/chat/sft/medqa/medqa_prompt/7B"

# add_model "qwen_2_5_3b_it_kd_qwen_prompt_medqa" "qwen2/chat/kd/medqa/qwen_prompt/3B"
# add_model "qwen_2_5_3b_it_kd_medqa_prompt_medqa" "qwen2/chat/kd/medqa/medqa_prompt/3B"

add_model "qwen_2_5_3b_it_kd_qwen_prompt_medqa_ckpt1" "qwen2/chat/kd/medqa/qwen_prompt/3B-ckpt1"
add_model "qwen_2_5_3b_it_kd_qwen_prompt_medqa_ckpt2" "qwen2/chat/kd/medqa/qwen_prompt/3B-ckpt2"
add_model "qwen_2_5_3b_it_kd_qwen_prompt_medqa_ckpt3" "qwen2/chat/kd/medqa/qwen_prompt/3B-ckpt3"

add_model "qwen_2_5_3b_it_kd_medqa_prompt_medqa_ckpt1"  "qwen2/chat/kd/medqa/medqa_prompt/3B-ckpt1"
add_model "qwen_2_5_3b_it_kd_medqa_prompt_medqa_ckpt2"  "qwen2/chat/kd/medqa/medqa_prompt/3B-ckpt2"
add_model "qwen_2_5_3b_it_kd_medqa_prompt_medqa_ckpt3"  "qwen2/chat/kd/medqa/medqa_prompt/3B-ckpt3"




for model_id in "${!ckpt_names[@]}"; do
    echo "=================================================="
    echo "Processing: $model_id"
    echo "=================================================="

    cd ${BASE_PATH_HARMBENCH}
    # PYTHONPATH=${BASE_PATH_HARMBENCH} VLLM_ENABLE_V1_MULTIPROCESSING=1 CUDA_VISIBLE_DEVICES=0,1 python ${BASE_PATH_HARMBENCH}/generate_completions.py \
    PYTHONPATH=${BASE_PATH_HARMBENCH} VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0,1 python ${BASE_PATH_HARMBENCH}/generate_completions.py \
        --model_name ${ckpt_names[$model_id]} \
        --models_config_file ${BASE_PATH_HARMBENCH}/configs/model_configs/models.yaml \
        --test_cases_path $test_cases_path \
        --behaviors_path $behaviors_path \
        --save_path $base_dir/completions/${output_dirs[$model_id]}.json \
        --max_new_tokens $max_new_tokens \
        --generate_with_vllm 

    # 2. Run judgment 
    PYTHONPATH=${BASE_PATH_HARMBENCH} VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=1 python evaluate_completions.py \
        --cls_path "cais/HarmBench-Llama-2-13b-cls" \
        --behaviors_path $behaviors_path \
        --completions_path $base_dir/completions/${output_dirs[$model_id]}.json \
        --save_path $base_dir/results/${output_dirs[$model_id]}_evaluation.json              
done