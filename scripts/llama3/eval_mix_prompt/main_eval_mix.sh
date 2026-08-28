BASE_PATH="/home/minillm_ai_safety"
LOG_FILE="evaluation_ckpt_mix_prompting_effect.log"

# Mirror all script output to both the terminal/tmux pane and the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

# DOLLY_PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama3"
LLAMA_PROMPT_DATA_DIR="${BASE_PATH}/processed_data/llama3/dolly/full/llama3-chat"

# For Student =  Dolly Prompt
PEFT_CKPT_NAME="sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424"


# For Teacher = LLama Prompt [e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890]
PEFT_CKPT_NAME="sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890"

# For Teacher =  Dolly Prompt
PEFT_CKPT_NAME="sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424"


###### Prompt variation begins for KD 
# Teacher = LLama Prompt, KD used Prompt = Dolly prompt
PEFT_CKPT_NAME="kd_dolly_prompt_teacher_llama_prompt/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424"

# Prompt variation begins for KD
# Teacher = Dolly Prompt, KD used Prompt = Dolly prompt
PEFT_CKPT_NAME="kd_dolly_prompt_teacher_dolly_prompt/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424"

# Prompt variation begins for KD
# Teacher = Dolly Prompt, KD used Prompt = LLama Prompt
PEFT_CKPT_NAME="kd_llama_prompt_teacher_dolly_prompt/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424"

# Prompt variation begins for KD
# Teacher = LLama Prompt, KD used Prompt = LLama Prompt
PEFT_CKPT_NAME="kd_llama_prompt_teacher_llama_prompt/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/"
PEFT_CKPT="/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/"


# PEFT_CKPT_NAME|PEFT_CKPT
VARIATIONS=(
# "kd_teacher_llama_kd_dolly|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/dolly-prompt/teacher_llama/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260103_034844/1068"
"mixing_0.8_kd_v2|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_022553/712"
"mixing_0.2_kd_v2|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_115557/1246"
"mixing_0.5_kd_v2|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260215_191116/890"
## 2ND ROUND EXPERIMENTS
# "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424|/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424"
# "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246|/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246"


## 3rd round experiments
# "gemma-2-2b-it|/media/drive2/models/gemma-models/gemma-2-2b-it"

## 4TH ROUND experiments
# "llama-3.2-3b-it|/media/drive2/models/llama-models/Llama-3.2-3B-Instruct" [this needs to be done]



## FOr base model evaluation
# "llama-3.2-3b-it|/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
# "llama-3.1-8b-it|/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"

#### For mixed prompt experiments.
# "kd/chat/8-3B/teacher-chat-data-dir/student-nonchat-data-dir/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260101_224450/890|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/teacher-chat-data-dir/student-nonchat-data-dir/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260101_224450/890"
# "kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt-test/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_021124/890|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/teacher-chat-data-dir/student-nonchat-data-dir/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt-test/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_021124/890"


### For instruction following experiments:

# "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246"
# "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424"
# "llama-3.2-3b-it|/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"


### just testing

# "kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_160837/890|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_160837/890"
# "kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_220635/890|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_220635/890"
### For meta math experiments

# "llama-3.2-3b-it|/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
# "llama-3.1-8b-it|/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"
# "kd/chat/8-3B/meta-math/meta-math-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/meta-math/meta-math-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251228_030924/1827/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251229_130758/3654|/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/meta-math/meta-math-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/meta-math/meta-math-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251228_030924/1827/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251229_130758/3654"
# "llama-3.1-8b-it|/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"
## ----------------- SFT 1B----------------
# "sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424|/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/1B/dolly-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_203531/1424"

# # ---------------- SFT 7B----------------
# "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424|/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424"

# "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890|/media/drive2/distillation/safety-utility-pg/results/llama2/train/sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890"

# # -------- KD: Teacher = llama, Prompt = dolly --------
# "kd_dolly_prompt_teacher_llama_prompt/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424|/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_115502/1424"

# # -------- KD: Teacher = dolly, Prompt = dolly --------
# "kd_dolly_prompt_teacher_dolly_prompt/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424|/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/dolly-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_065653/1424"

# # -------- KD: Teacher = dolly, Prompt = llama --------
# "kd_llama_prompt_teacher_dolly_prompt/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424|/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_091805/1424"

# # -------- KD: Teacher = llama, Prompt = llama --------
# "kd_llama_prompt_teacher_llama_prompt/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/1068|/media/drive2/distillation/safety-utility-pg/results/llama2/train/kd/7-1B/llama-prompt/0.5/teacher-base/Llama-2-7b-chat-hf/teacher-adapter/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251219_143148/1068"
)


# echo "Starting Evaluation ......"
# CUDA_VISIBLE_DEVICES=2,3,4,5 bash scripts/llama2/eval/run_eval.sh dolly-prompt $DOLLY_PROMPT_DATA_DIR $PEFT_CKPT_NAME $PEFT_CKPT
# CUDA_VISIBLE_DEVICES=2,3,4,5 bash scripts/llama2/eval/run_eval.sh llama-prompt $LLAMA_PROMPT_DATA_DIR $PEFT_CKPT_NAME $PEFT_CKPT




PROMPTS=("llama-prompt")
CKPT_NAME="llama-3.2-3b-it"
CKPT_PATH="/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
# CKPT_NAME="llama-3.1-8b-it"
# CKPT_PATH="/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"

echo "Starting Evaluation for ${CKPT_NAME}......"
for entry in "${VARIATIONS[@]}"; do
    IFS="|" read PEFT_CKPT_NAME PEFT_CKPT <<< "$entry"

        if [[ "${PEFT_CKPT_NAME}" == "llama-3.1-8b-it" || "${PEFT_CKPT_NAME}" == "sft_8b_it_llama_prompt_1068" || "${PEFT_CKPT_NAME}" == "sft_8b_it_dolly_prompt_1068" ]]; then
            CKPT_NAME="llama-3.1-8b-it"
            CKPT_PATH="/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"
        fi

        {
            echo "----------------------------------------"
            echo "Checkpoint  : $PEFT_CKPT_NAME"
            echo "Peft ckpt name : $PEFT_CKPT_NAME"
            echo "Peft ckpt path : $PEFT_CKPT"
            echo "----------------------------------------"
        }

        CUDA_VISIBLE_DEVICES=0,1 bash scripts/llama3/eval_mix_prompt/eval/run_eval.sh "$PEFT_CKPT_NAME" "$PEFT_CKPT" "$CKPT_NAME" "$CKPT_PATH"
done
