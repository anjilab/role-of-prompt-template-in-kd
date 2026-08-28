#!/bin/bash

BASE_PATH="/home/luffy/projects/ai/LMOps/minillm"

# Declare associative arrays
declare -A model_paths      
declare -A model_leaf_dir      
declare -A model_checkpoint_name      
declare -A model_adapter      
    
# Llama2Adapter
# DollyV2Adapter

add_model() {
    local ckpt_path="$1"        
    local output_dir="$2"               
    local adapter="$3"     

    local ckpt_name=$(basename "$output_dir")                   
    local ckpt_leaf_dir=$(dirname "$output_dir")           

    model_paths["$output_dir"]="$ckpt_path"
    model_leaf_dir["$output_dir"]="$ckpt_leaf_dir"
    model_checkpoint_name["$output_dir"]="$ckpt_name"
    model_adapter["$output_dir"]="$adapter"
}


num_gpus_total=2

# === Register all your models here ===

# TiniyLlama_v1.1 Base Model
# add_model "/media/drive2/models/llama-models/tiny-llama-1.1B" "llama2/base-tinyllama1B"   "DollyV2Adapter"

# # TiniyLlama_v1.1 SFT
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/sft/tinyllama-1b-sft" "llama2/sft-tinyllama1B"  "DollyV2Adapter"
# TiniyLlama_v1.1 SFT
# add_model "results/llama2/merged/sft/1B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251214_233732/1068" "llama2/sft-tinyllama1B"  "DollyV2Adapter"

# TiniyLlama_v1.1 Base Model - LLAMA Adapter
# add_model "checkpoints/llama2/TinyLlama_v1.1" "llama2/base-tinyllama1B-llama-adapter"   "Llama2Adapter"

# # TiniyLlama_v1.1 SFT - LLAMA Adapter
# add_model "results/llama2/merged/sft/1B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251214_233732/1068" "llama2/sft-tinyllama1B-llama-adapter"  "Llama2Adapter"

# # Llama 7B Base Model
# add_model "/media/drive2/models/llama-models/Llama-2-7b-chat-hf" "llama2/base-llama7B" "Llama2Adapter"

# # Llama 7B SFT - Dolly Prompt
# add_model "results/llama2/merged/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068" "llama2/sft-llama7B-dolly-prompt-with-dolly-adapter" "DollyV2Adapter"

# # Llama 7B SFT - Llama Prompt
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/sft/llama-7b-it" "llama2/sft-llama7B-llama-prompt" "Llama2Adapter"
# add_model "results/llama2/merged /sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068" "llama2/sft-llama7B-llama-prompt-with-dolly-adapter" "DollyV2Adapter"

# # Llama 7B SFT - Dolly Prompt
# add_model "results/llama2/merged/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068" "llama2/sft-llama7B-dolly-prompt-with-llama-adapter-v2" "Llama2Adapter"

# # Llama 7B SFT - Llama Prompt
# add_model "results/llama2/merged/sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068" "llama2/sft-llama7B-llama-prompt-with-llama-adapter-v2" "Llama2Adapter"

# add_model "results/llama2/merged/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/534" "llama2/sft-llama7B-dolly-534ckpt-prompt-llama-adapter-v2" "Llama2Adapter"
# add_model "results/llama2/merged/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/178" "llama2/sft-llama7B-dolly-178ckpt-prompt-llama-adapter-v2" "Llama2Adapter"

# # KD from Llama 2_7B(Dolly Prompt) to TinyLlama_1B - Dolly Prompt - 0.5
# add_model "results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251215_110135/1068" "llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-dolly-prompt"  "DollyV2Adapter"

# # KD from Llama 2_7B(LLama Prompt) to TinyLlama_1B - Dolly Prompt - 0.5
# add_model "results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068/e10-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/890" "llama2/kd-0.5-llama7B-llama-prompt-tinyllama1B-dolly-prompt" "DollyV2Adapter"


# # KD from Llama 2_7B(Dolly Prompt) to TinyLlama_1B - Dolly Prompt - 1.0
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/kd/fkl_rkl/tiny_llama_1B_1_0_seed_42_fkl" "llama2/kd-1.0-tiny_llama_1B_1_0_seed_42_fkl" "DollyV2Adapter"

# # KD from Llama 2_7B(LLama Prompt) to TinyLlama_1B - LLama Prompt - 0.5
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/kd/fkl_rkl/tiny_llama_1B_0_5_0_seed_42_fkl" "llama2/kd-0.5-tiny_llama_1B_0_5_0_seed_42_fkl" "DollyV2Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/kd/fkl_rkl/llama_prompt/tiny_llama_1B_0_5_seed_42_fkl_llama_prompt" "llama2/tiny_llama_1B_0_5_seed_42_fkl_llama_prompt" "Llama2Adapter"


# add_model "/media/drive2/models/llama-models/kc-academy/llama-2-1B-kd0.5-DTLP" "kc-academy/llama-2-1B-kd0.5-DTLP" "Llama2Adapter"
# add_model "/media/drive2/models/llama-models/kc-academy/llama-2-1B-kd0.5-LTLP" "kc-academy/llama-2-1B-kd0.5-LTLP" "Llama2Adapter"


# TiniyLlama_v1.1 Base Model
# add_model "/media/drive2/models/llama-models/tiny-llama-1.1B" "llama2/reproducibility/base-tinyllama1B_llama2adapte" "Llama2Adapter"
# add_model "/media/drive2/models/llama-models/tiny-llama-1.1B" "llama2/reproducibility/base-tinyllama1B_dollyv2adapter" "DollyV2Adapter"

# TiniyLlama_v1.1 sft dolly prompt
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424" "llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424" "llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_dollyv2adapter" "DollyV2Adapter"


# # Llama 7B Base Model
# add_model "/media/drive2/models/llama-models/Llama-2-7b-chat-hf" "llama2/reproducibility/base-llama7B_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/models/llama-models/Llama-2-7b-chat-hf" "llama2/reproducibility/base-llama7B_dollyv2adapter" "DollyV2Adapter"


# # Llama 7B SFT - Llama Prompt
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890" "llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890" "llama2/reproducibility/sft/7B/llama-prompt/890/verify_llama_prompt_sft_7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890" "llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_dollyv2adapter" "DollyV2Adapter"

# # Llama 7B SFT - Dolly Prompt
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424" "llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424" "llama2/reproducibility/sft/7B/dolly-prompt/1424/verify_dolly_prompt_sft_7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424" "llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_dollyv2adapter" "DollyV2Adapter"

# # Teacher - Dolly prompt, KD - Dolly prompt  - 0.5
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424" "llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424" "llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter" "DollyV2Adapter"


# # Teacher - Dolly prompt, KD - LLama prompt  - 0.5
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424" "llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424" "llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter" "DollyV2Adapter"


# # Teacher - LLama prompt, KD - Dolly prompt  - 0.5
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424" "llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424" "llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter" "DollyV2Adapter"


# # Teacher - LLama prompt, KD - LLama prompt  - 0.5
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/1068" "llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/1068" "llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter" "DollyV2Adapter"


# add_model "/media/drive2/distillation/hf-models-testing/kc-academy/luffy-sft-llama-7B-LP" "llama2/reproducibility/sft/7B/llama-prompt/kc-academy/luffy-sft-llama-7B-LP" "Llama2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1068" "llama2/reproducibility/sft/7B/dolly-prompt/1068-ckpt-llama2adapter" "Llama2Adapter"



# KD from Llama 2_7B(Dolly Prompt) to TinyLlama_1B - LLama Prompt - 0.5
# add_model "results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251218_000921/1424" "llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-llama-prompt" "Llama2Adapter"

# KD from Llama 2_7B(Zeroshot) to TinyLlama_1B - LLama Prompt - 0.5
# add_model "results/llama2/merged/kd/7-1B/llama-prompt/0.5/teacher-zeroshot/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251218_042123/712" "llama2/kd-0.5-llama7B-zeroshot-tinyllama1B-llama-prompt" "Llama2Adapter"

# try

#  Llama 7B Base Model
# add_model "checkpoints/llama2/7b-chat-base" "v2/llama2/v2/base-llama7B" "Llama2Adapter"

# running all 7B LLAMA dolly-prompt checkpoints with LLAMA adapter
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "results/llama2/merged/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/${ckpt}" "llama2/v2/sft-llama7B-dolly-${ckpt}-llama-adapter" "Llama2Adapter"
# done

# #  Llama 7B Base Model
# add_model "checkpoints/llama2/7b-chat-base" "llama2/v3/base-llama7B" "Llama2Adapter"

# # running all 7B LLAMA llama-prompt checkpoints with LLAMA adapter
# for ckpt in 178 356 534 712 890 1068 1246 1424 1602 1780; do
#     add_model "results/llama2/merged/sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/${ckpt}" "llama2/v3/sft-llama7B-llama-${ckpt}-llama-adapter" "Llama2Adapter"
# done

# add_model "checkpoints/llama2/TinyLlama_v1.1" "llama2/base-tinyllama1B"   "DollyV2Adapter"
#  Llama 1.1B Base Model
# add_model "checkpoints/llama2/TinyLlama-1.1B-Chat-v1.0" "llama2/chat/base-tinyllama1.1B-chat" "Llama2Adapter"

# #  Llama 1.1B Base Model Chat
# add_model "results/llama2/merged/sft/chat/1B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251220_185453/1068" "llama2/chat/sft-tinyllama1B-dolly-prompt" "Llama2Adapter"


## gemma
# add_model "checkpoints/gemma/2/gemma-2-2b" "gemma2/base-gemma2B" "GemmaAdapter"
# add_model "results/gemma2/merged/sft/2B/dolly-prompt/e8-bs16-lr2e-05-G4-N1-NN1-lora-8-16-0.1/20251218_190731/1068" "gemma2/sft-gemma2B-dolly-prompt" "GemmaAdapter"
# add_model "/media/drive2/models/llama-models/Llama-2-7b-chat-hf" "llama2/v3/base-llama7B" "Llama2Adapter"


# add_model "/media/drive2/models/llama-models/tiny-llama-1.1B-chat" "llama2/chat/tinyllama-base" "Llama2Adapter"
# # running all 7B LLAMA dolly-prompt checkpoints with LLAMA adapter
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     # add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/${ckpt}" "llama2/v2/sft-llama7B-dolly-${ckpt}-llama-adapter" "Llama2Adapter"
#     add_model "/media/drive2/distillation/safety-utility-pg/results/llama2/merged/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/${ckpt}" "llama2/v3/sft-llama7B-llama-${ckpt}-llama-adapter" "Llama2Adapter"

# done
# add_model "/media/drive2/models/llama-models/Llama-2-7b-chat-hf" "llama2/hypothesis/teacher_base" "Llama2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/sft/llama-7b-it" "llama2/hypothesis/teacher_sft" "Llama2Adapter"

# add_model "/media/drive2/models/llama-models/tiny-llama-1.1B" "llama2/chat/tinyllama_base" "DollyV2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/sft/tinyllama-1b-sft" "llama2/chat/tinyllama_sft" "DollyV2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/kd/fkl_rkl_safety/tiny_llama_1B_0_5_0_seed_42_fkl_safety" "llama2/hypothesis/fkl" "DollyV2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama/merged/kd/fkl_rkl_safety/tiny_llama_1B_0_5_0_seed_42_rkl_safety" "llama2/hypothesis/rkl" "DollyV2Adapter"

# gemma 2B
# add_model "/media/drive2/models/gemma-models/gemma-2-2b-it" "gemma2/chat/v2/base-gemma2B-gaamma-adapter" "GemmaAdapter"
# add_model "/media/drive2/models/gemma-models/gemma-2-9b-it" "gemma2/chat/v2/base-gemma9B-gaamma-adapter" "GemmaAdapter"
#  # All gemma 2B checkpoints - dolly prompt
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/${ckpt}" "gemma2/chat/sft-gemma2B-dolly-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done

#  # All gemma 2B checkpoints - gemma prompt
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/${ckpt}" "gemma2/chat/sft-gemma2B-gemma-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done

#  # All gemma 2B checkpoints - dolly prompt
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/${ckpt}" "gemma2/chat/v2/sft-gemma9B-gemma-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done
#  # All gemma 2B checkpoints - dolly prompt - 1e-5
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/${ckpt}" "gemma2/chat/1e-5/sft-gemma2B-dolly-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done
#  # All gemma 2B checkpoints - dolly prompt - 3e-5
# for ckpt in 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/${ckpt}" "gemma2/chat/3e-5/sft-gemma2B-dolly-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done


## Gemma 2B checkpoints - dolly promt - 0.0002
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/${ckpt}" "gemma2/chat/2e-4/sft-gemma2B-dolly-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done

# # Gemma 2B checkpoints - dolly prompt - 0.0001
# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/${ckpt}" "gemma2/chat/1e-4/sft-gemma2B-dolly-${ckpt}-gaamma-adapter" "GemmaAdapter"
# done


### Gemma 9B checkpoints KD dolly prompt, teacher dolly prompt
# for ckpt in  178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/${ckpt}" "gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-${ckpt}-gemma-adapter" "GemmaAdapter"
# done

### Gemma 9B checkpoints  KD gemma teacher, teacher gemma prompt 
# for ckpt in  178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/${ckpt}" "gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-${ckpt}-gemma-adapter" "GemmaAdapter"
# done

# 3b-it sft
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1424" "gemma2/chat/v2/sft/2B/dolly_prompt/" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1068" "gemma2/chat/v2/sft/2B/gemma_prompt/" "GemmaAdapter"


### Llama3.2 version
# add_model "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct" "llama3/chat/base/3B" "Llama3Adapter"
# add_model "/media/drive2/models/llama-models/Llama-3.1-8B-Instruct" "llama3/chat/base/8B" "Llama3Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068" "llama3/chat/sft/8B/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068" "llama3/chat/sft/8B/llama-prompt" "Llama3Adapter"

# 3b-it sft
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_170116/890" "llama3/chat/sft/v2/3B/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_200923/1246" "llama3/chat/sft/v2/3B/llama-prompt" "Llama3Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246" "llama3/chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424"  "llama3/chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt" "Llama3Adapter"

# # running all gemma KD 9B-2B IT - 0.5 - gemma prompt with Gemma adapter
# for ckpt in 1424; do
#     add_model "results/gemma2/merged/kd/chat/9-2B/gemma-prompt/0.5/e8-bs8-lr2e-05-G8-N1-NN1-kd0.5-lora-8-16-0.1/20251222_200314/${ckpt}" "gemma2/chat/kd/0.5/9-2B-gemma-${ckpt}-gemma-adapter" "GemmaAdapter"
# done


# # llama_3_3B Base Model
# add_model "checkpoints/llama3/Llama-3.2-3B-Instruct" "llama3/base-3B"   "Llama3Adapter"

# # llama_3_8B Base Model
# add_model "checkpoints/llama3/Llama-3.1-8B-Instruct" "llama3/base-8B"   "Llama3Adapter"


#### Dec 30 Running models
# add_model "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct" "llama3/base-3B"   "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246" "llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424"  "llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt" "Llama3Adapter"


# add_model "/media/drive2/models/gemma-models/gemma-2-2b-it" "gemma2/seed/v2/base-gemma2B-gemma-adapter" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424" "gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246" "gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt" "GemmaAdapter"
# SEEDS=(10 20 30 40 50)

#### Jan 3 for sft teacher models
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068" "llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068" "llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424" "gemma2/seed/v2/sft_gemma_prompt_1424_9b-it" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712" "gemma2/seed/v2/sft_dolly_prompt_712_9b-it" "GemmaAdapter"



### Gemma 9B checkpoints KD dolly prompt, teacher dolly prompt
# for ckpt in  178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/${ckpt}" "gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-${ckpt}-gemma-adapter" "GemmaAdapter"
# done


# for ckpt in 178 356 534 712 890 1068 1246 1424; do
#     add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/${ckpt}" "llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-${ckpt}-llama-adapter" "Llama3Adapter"
# done

# For review RKL

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/rkl/0.5/20260209_174940/890" "llama3/chat/v2/8-3B/rkl/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/rkl/0.5/20260209_231238/890" "llama3/chat/v2/8-3B/rkl/llama-prompt" "Llama3Adapter"
 
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260210_144627/1424" "gemma2/seed/v2/rkl/dolly-prompt" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260212_100241/890" "gemma2/seed/v2/rkl/gemma-prompt" "GemmaAdapter"


# add_model "/media/drive2/models/llama-models/Llama-3.2-1B-Instruct"  "llama3/metamath/base/1B" "Llama3Adapter"
# add_model "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct" "llama3/metamath/base/3B" "Llama3Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/fkl+rkl/0.5/20260218_060855/1246" "llama3/chat/v2/8-3B/fkl+rkl/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/fkl+rkl/0.5/20260218_012550/712" "llama3/chat/v2/8-3B/fkl+rkl/llama-prompt" "Llama3Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_022553/712" "llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_115557/1246" "llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260215_191116/890" "llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5" "Llama3Adapter"


# add_model "/media/drive2/models/llama-models/Llama-3.1-8B-Instruct" "llama3/seed/v2/latest/base-8B-it" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068" "llama3/chat/sft/8B/latest/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068" "llama3/chat/sft/8B/latest/llama-prompt" "Llama3Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/dolly-prompt/teacher-criscross-evaluation/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260406_223602/712" "llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/teacher-criscross-evaluation/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260407_143936/534" "llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt" "Llama3Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.1/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260416_124134/890" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_115557/1246" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.4/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260418_022406/890" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260215_191116/890" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.6/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260418_102456/1424" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt" "Llama3Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_022553/712" "llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt" "Llama3Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/dolly-prompt/teacher-criscross-evaluation/teacher_gemma_student_dolly/1246" "gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt" "GemmaAdapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/teacher-criscross-evaluation/teacher_dolly_student_gemma/1246" "gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt" "GemmaAdapter"

    
# add_model "/media/scratch/safety-utility-pg/results/gemma2/merged/kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.4/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260422_151350/890" "gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt" "GemmaAdapter"
# add_model "/media/scratch/safety-utility-pg/results/gemma2/merged/kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260422_200815/1246" "gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt" "GemmaAdapter"
# add_model "/media/scratch/safety-utility-pg/results/gemma2/merged/kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.6/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260423_153826/1424" "gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt" "GemmaAdapter"
# add_model "/media/scratch/safety-utility-pg/results/gemma2/merged/kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.8/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260423_202051/890" "gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt" "GemmaAdapter"

# add_model "/media/drive2/models/llama-models/Llama-3.2-1B-Instruct"  "llama3/metamath/base/1B" "Llama3Adapter"

# add_model "/media/drive2/models/gemma-models/gemma-2-9b-it" "gemma2/seed/v2/base/gemma-9B-gemma-adapter" "GemmaAdapter"

# add_model "/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct"  "qwen2/base/3B" "Qwen2Adapter"
# add_model "/media/drive2/models/qwen-models/Qwen2.5-7B-Instruct"  "qwen2/base/7B" "Qwen2Adapter"


# add_model "/media/drive2/models/llama-models/Llama-3.2-1B-Instruct" "llama3/chat/base/1B" "Llama3Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/med-qa/llama-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.05/20260515_194048/936" "llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16" "Llama3Adapter" # sft llama prompt 3b-it in med-qa dataset
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/med-qa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.05/20260515_221520/624" "llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16" "Llama3Adapter" # sft med prompt 3b-it in med-qa dataset

# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/med-qa/llama-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05/20260512_221743/312" "llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt" "Llama3Adapter" # sft llama prompt 3b-it in med-qa dataset
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/sft/llama3b-it/med-qa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05/20260512_003339/624" "llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt" "Llama3Adapter" # sft med prompt 3b-it in med-qa dataset
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/llama-prompt/med-qa/0.5/teacher-base/llama-8B/teacher-adapter/sft_llama8b-it_med-qa_llama-prompt_e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05_20260510_103454_624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260510_220342/624" "llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt" "Llama3Adapter" # kd 0.5 med-qa llama prompt 3b-it
# add_model "/media/drive2/distillation/safety-utility-pg/results/llama3/merged/kd/chat/8-3B/medqa-prompt/med-qa/0.5/teacher-base/llama-8B/teacher-adapter/sft_llama8b-it_med-qa_medqa-prompt_e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05_20260511_192134_312/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260512_172436/312" "llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt" "Llama3Adapter" # kd ₀.5 med-qa med prompt 3b-it

# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/med-qa/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft_gemma9b-it_med-qa_gemma2-prompt_e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05_20260513_191623_936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.05/20260517_024942/624" "gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt" "GemmaAdapter" # kd 0.5 med-qa gemma prompt 9b-it
# add_model "/media/drive2/distillation/safety-utility-pg/results/gemma2/merged/kd/chat/9-2B/gemma-prompt/med-qa/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft_gemma9b-it_med-qa_medqa-prompt_e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05_20260513_040514_312/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.05/20260517_231800/936" "gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt" "GemmaAdapter" # kd 0.5 med-qa med prompt 9b-it

# add_model "/media/drive2/models/qwen-models/Qwen2.5-7B-Instruct"  "qwen2/base/7B" "Qwen2Adapter"
# add_model "/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct"  "qwen2/base/3B" "Qwen2Adapter"
# # add_model "/media/drive2/models/qwen-models/Qwen2.5-1.5B-Instruct" "qwen2/base/1B" "Qwen2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_032105/1424" "qwen2/seed/v2/sft/dolly/qwen-prompt/3B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_112846/1068" "qwen2/seed/v2/sft/dolly/dolly-prompt/3B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068" "qwen2/seed/v2/sft/dolly/qwen-prompt/7B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246" "qwen2/seed/v2/sft/dolly/dolly-prompt/7B" "Qwen2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_085107/1424" "qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_024037/1424" "qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt" "Qwen2Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_021754/624" "qwen2/seed/v2/sft/medqa/qwen-prompt/3B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-3B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_053035/936" "qwen2/seed/v2/sft/medqa/medqa-prompt/3B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936" "qwen2/seed/v2/sft/medqa/qwen-prompt/7B" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624" "qwen2/seed/v2/sft/medqa/medqa-prompt/7B" "Qwen2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/624" "qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/936" "qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt" "Qwen2Adapter"

# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/312" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/624" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/936" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3" "Qwen2Adapter"


# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/312" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/624" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/936" "qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3" "Qwen2Adapter"


add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/rkl/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260524_133641/1068" "qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt" "Qwen2Adapter"
# add_model "/media/drive2/distillation/safety-utility-pg/results/qwen2/merged/rkl/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260523_194624/534" "qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt" "Qwen2Adapter"
# SEEDS=(10 20 30)
SEEDS=(30)




for seed in "${SEEDS[@]}"; do
  echo "=================================================="
  echo "Running with SEED = $seed"
  echo "=================================================="
    for model_id in "${!model_paths[@]}"; do
        echo "=================================================="
        echo "Processing: $model_id"
        echo "Model Path: ${model_paths[$model_id]}"
        echo "Leaf Dir: ${model_leaf_dir[$model_id]}"
        echo "Checkpoint Name: ${model_checkpoint_name[$model_id]}"
        echo "Adapter: ${model_adapter[$model_id]}"
        echo "=================================================="
        seed_name="seed${seed}"
        formatted_model_id="${model_id}/${seed_name}"
        # 1. Generate answers
        CUDA_VISIBLE_DEVICES=0,1 python sorry-bench/gen_model_answer_vllm.py \
            --bench-name sorry_bench \
            --model-path "${model_paths[$model_id]}" \
            --model-id "${formatted_model_id}"  \
            --template-adapter "${model_adapter[$model_id]}" \
            --num-gpus-total $num_gpus_total \
            --num-gpus-per-model $num_gpus_total \
            --seed $seed \
            # --data-mutation logical_appeal

        # 2. Run judgment 

        # output_dir/model_id = llama3/base-3B
            # prefix_dir = llama3 
            # checkpoint_name = base-3B
        # formatted_model_id="${model_id}/seed${seed}" = llama3/base-3B/seed10

        CUDA_VISIBLE_DEVICES=1 python sorry-bench/gen_judgment_safety_vllm.py \
            --model-list "${seed_name}" \
            --answer-dir-prefix "${model_id}/" \
            --judgement-path "${formatted_model_id}"  \
            # --data-mutation logical_appeal                  
    done
done