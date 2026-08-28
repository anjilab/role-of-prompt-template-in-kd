#!/bin/bash


BASE_PATH="/home/minillm_ai_safety"
VENV_PATH="${BASE_PATH}/walledeval/.venv"
PYTHON="${VENV_PATH}/bin/python"
EVAL_SCRIPT="${BASE_PATH}/walledeval/eval_safety.py"
RESULTS_BASE="${BASE_PATH}/walledeval/results"
JUDGE_VERSION=3

source "${VENV_PATH}/bin/activate"

# Datasets
declare -A DATASETS=(
    [advbench]="AdvBench|${BASE_PATH}/data/walledai/AdvBench/data/train-00000-of-00001.parquet|prompt"
    [jailbreakbench]="JailbreakBench|${BASE_PATH}/data/walledai/JailbreakBench/data/train-00000-of-00001.parquet|prompt"
    # [xstest]="XSTest|${BASE_PATH}/data/walledai/XSTest/data/train-00000-of-00001.parquet|prompt"
)

# Models
declare -a MODELS=(


    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_170116/890|llama3/chat/sft/v2/3B/dolly-prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_200923/1246|llama3/chat/sft/v2/3B/llama-prompt"


    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1424|gemma2/chat/v2/sft/2B/dolly_prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1068|gemma2/chat/v2/sft/2B/gemma_prompt" 


    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/rkl/0.5/20260209_174940/890|llama3/chat/v2/8-3B/rkl/dolly-prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/rkl/0.5/20260209_231238/890|llama3/chat/v2/8-3B/rkl/llama-prompt"

    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260210_144627/1424|gemma2/seed/v2/rkl/dolly-prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260212_100241/890|gemma2/seed/v2/rkl/gemma-prompt" 

    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/fkl+rkl/0.5/20260218_060855/1246|llama3/chat/v2/8-3B/fkl+rkl/dolly-prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/fkl+rkl/0.5/20260218_012550/712|llama3/chat/v2/8-3B/fkl+rkl/llama-prompt"


    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_170116/890|llama3/chat/sft/v2/3B/dolly-prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_200923/1246|llama3/chat/sft/v2/3B/llama-prompt"


    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1424|gemma2/chat/v2/sft/2B/dolly_prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1068|gemma2/chat/v2/sft/2B/gemma_prompt" 


 
    # "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct|llama3/base_llama3.2-3b-it"
  
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068|llama3/sft_llama_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068|llama3/sft_dolly_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246|llama3/kd_dolly_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424|llama3/kd_llama_prompt"

    # "/media/drive2/models/gemma-models/gemma-2-2b-it|gemma2/base_gemma2-2b-it"
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424|gemma2/sft_gemma_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712|gemma2/sft_dolly_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424|gemma2/kd_dolly_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246|gemma2/kd_gemma_prompt"
    

    # "/media/drive2/models/qwen-models/Qwen2.5-7B-Instruct|qwen2/base_qwen2.5-7b-it"  
    # "/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct|qwen2/base_qwen2.5-3b-it"  

    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_032105/1424|qwen2/3B/dolly/qwen-prompt/sft_qwen_prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_112846/1068|qwen2/3B/dolly/dolly-prompt/sft_dolly_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068|qwen2/7B/dolly/qwen-prompt/sft_qwen_prompt" 
    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246|qwen2/7B/dolly/dolly-prompt/sft_dolly_prompt"

    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_085107/1424|qwen2/kd/dolly/dolly-prompt/kd_dolly_prompt" 
    "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_024037/1424|qwen2/kd/dolly/qwen-prompt/kd_qwen_prompt"  

    ### FOR MEDQA

    "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_021754/624|qwen2/3B/medqa/qwen-prompt/sft_qwen_prompt" 
    "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_053035/936|qwen2/3B/medqa/medqa-prompt/sft_medqa_prompt"
    "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936|qwen2/7B/medqa/qwen-prompt/sft_qwen_prompt" 
    "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624|qwen2/7B/medqa/medqa-prompt/sft_medqa_prompt"

    ## MEDQA-KD
    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/624|qwen2/kd/medqa/medqa-prompt/kd_medqa_prompt"
    # "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/936|qwen2/kd/medqa/qwen-prompt/kd_qwen_prompt"
   )

# Run evaluation on a GPU
run_eval() {
    local model_path="$1"
    local result_subdir="$2"
    local dataset_key="$3"
    local gpu="$4"
    local seed="$5"
    
    IFS='|' read -r dataset_name data_path prompt_col <<< "${DATASETS[$dataset_key]}"
    
    output_dir="${RESULTS_BASE}/${result_subdir}/${dataset_key}/seed_${seed}"
    mkdir -p "$output_dir"
    
    echo "[GPU $gpu] Evaluating $(basename $result_subdir) on $dataset_name (seed=$seed)"
    
    CUDA_VISIBLE_DEVICES="$gpu" ${PYTHON} "${EVAL_SCRIPT}" \
        --model_name_or_path "${model_path}" \
        --data_path "${data_path}" \
        --dataset_name "${dataset_name}" \
        --prompt_column "${prompt_col}" \
        --output_dir "${output_dir}" \
        --judge_version "${JUDGE_VERSION}" \
        --seed "$seed" 2>&1 | sed "s/^/[GPU $gpu] /"
}

# Main loop - job queue with GPU management
declare -a SEEDS=(10 20)
declare -a AVAILABLE_GPUS=(0 1)
declare -A GPU_PIDS=()  # Track PIDs for each GPU
declare -a JOB_QUEUE=()

# Build job queue
for seed in "${SEEDS[@]}"; do
    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_path result_subdir <<< "$model_entry"
        for dataset_key in advbench jailbreakbench; do
        # for dataset_key in advbench; do

            JOB_QUEUE+=("$model_path|$result_subdir|$dataset_key|$seed")
        done
    done
done

echo "Total jobs to run: ${#JOB_QUEUE[@]}"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo ""

# Process job queue
job_idx=0

while [ $job_idx -lt ${#JOB_QUEUE[@]} ] || [ ${#GPU_PIDS[@]} -gt 0 ]; do
    # Check which GPUs are available
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        pid="${GPU_PIDS[$gpu]:-}"
        
        # Check if GPU has a running job
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            # GPU job finished
            wait "$pid" 2>/dev/null
            echo "✓ GPU $gpu completed a job"
            unset GPU_PIDS[$gpu]
        fi
        
        # If GPU is free and jobs remaining, assign next job
        if [ -z "${GPU_PIDS[$gpu]:-}" ] && [ $job_idx -lt ${#JOB_QUEUE[@]} ]; then
            job="${JOB_QUEUE[$job_idx]}"
            IFS='|' read -r model_path result_subdir dataset_key seed <<< "$job"
            
            echo "[GPU $gpu] Assigning job $((job_idx+1))/${#JOB_QUEUE[@]}: $(basename $result_subdir) - $dataset_key - seed=$seed"
            
            run_eval "$model_path" "$result_subdir" "$dataset_key" "$gpu" "$seed" &
            GPU_PIDS[$gpu]=$!
            ((job_idx++))
        fi
    done
    
    sleep 2
done

wait
echo "Done! All jobs completed!"

wait
echo "Done!"
