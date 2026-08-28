import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig

BASE_PATH="/home/minillm_ai_safety"
DRIVE_PATH="/media/drive2/distillation/safety-utility-pg"
# DRIVE_PATH="/media/scratch/safety-utility-pg"


def merge_model(base_model_path, adapter_path, merged_model_path):

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Load the tokenizer (optional but recommended)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Wrap the base model with the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge the LoRA into the base model and unload the adapter
    merged_model = peft_model.merge_and_unload()

    # Save the merged model (now a full standalone model)
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    if "qwen" in base_model_path.lower():
        base_tokenizer_config = f"{base_model_path}/tokenizer_config.json"
        merged_tokenizer_config = f"{merged_model_path}/tokenizer_config.json"
        shutil.copy2(base_tokenizer_config, merged_tokenizer_config)
    

# FKL-RKL full model conversion
models = {  
    "gemma2BChat":[
        # {"adapter_path": "kd/chat/9-2B/dolly-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260210_144627/1424"},
        # {"adapter_path": "kd/chat/9-2B/gemma-prompt/rkl/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260212_100241/890"}
        # All gemma 2B checkpoints - dolly prompt
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/178"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/356"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/534"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/712"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/890"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1068"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1246"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_021942/1424" # high val
        # },
        #  # All gemma 2B checkpoints - gemma prompt
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/178"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/356"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/534"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/712"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/890"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1068" # high val
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1246"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/gemma-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251221_102713/1424"
        # },
         # All gemma 2B checkpoints - dolly prompt - 1e-5
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/178"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/356"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/534"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/712"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/890"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/1068"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/1246"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr1e-05-G8-N4-NN1-lora-8-16-0.1/20251221_234610/1424"
        #  },
         
        #   # All gemma 2B checkpoints - dolly prompt - 3e-5
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/178"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/356"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/534"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/712"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/890"
        #  },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/1068"
        #  },
        #   {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/1246"
        #  },
        #    {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr3e-05-G8-N4-NN1-lora-8-16-0.1/20251222_093046/1424"
        #  }
        #  for lr 1e-4
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/178"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/356"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/534"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/712"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/890"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/1068"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/1246"
        # },
        #  {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0001-G8-N4-NN1-lora-8-16-0.1/20251222_120455/1424"
        #  },
        # #  for lr 2e-4
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/178"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/356"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/534"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/712"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/890"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/1068"
        # },
        # {
        #     "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/1246"
        # },
        # {
        #      "adapter_path": "sft/chat/gemma-2-2B/dolly-prompt/e8-bs2-lr0.0002-G8-N4-NN1-lora-8-16-0.1/20251222_142544/1424"
        # } 
        
        # # gemma-9b-it kd teacher dolly-prompt, kd = dolly-prompt
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/178"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/356"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/534"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/712"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/890"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1068"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1246"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424" # high val
        # },
        
        
        
        # # gemma-9b-it kd teacher gemma-prompt, kd = gemma-prompt
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/178"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/356"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/534"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/712"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/890"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1068"
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246" #high val
        # },
        # {
        #     "adapter_path": "kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1424"
        # }, 
        
        
        # crisscross evaluation
            # {
            #     "adapter_path": "kd/chat/9-2B/dolly-prompt/teacher-criscross-evaluation/teacher_gemma_student_dolly/1246"
            # },
            # {
            #     "adapter_path": "kd/chat/9-2B/gemma-prompt/teacher-criscross-evaluation/teacher_dolly_student_gemma/1246"
            # },
            
            # Prompt mixing 
            
            # {
            #     "adapter_path": "kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.8/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260423_202051/890"
            # },
            # {
            #    "adapter_path": "kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.6/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260423_153826/1424" 
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260422_200815/1246"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/gemma-prompt/mixing/v2/kd/0.5/curriculum_0.4/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260422_151350/890"
            # },
            
            
            # medqa prompt
            # {
            #     "adapter_path": ""
            # }
            # {
            #     "adapter_path": "kd/chat/9-2B/gemma-prompt/med-qa/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft_gemma9b-it_med-qa_gemma2-prompt_e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05_20260513_191623_936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.05/20260517_024942/624"
            # },
            # {
            #     "adapter_path": "kd/chat/9-2B/gemma-prompt/med-qa/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft_gemma9b-it_med-qa_medqa-prompt_e3-bs1-lr0.0001-G32-N4-NN1-lora-16-32-0.05_20260513_040514_312/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.05/20260517_231800/936"
            # }
    
               
    ],
    "gemma9BChat":[
        # gemma 9B SFT
        # {
        #     "adapter_path": "sft/9B/dolly-prompt/e8-bs8-lr2e-05-G8-N1-NN1-lora-8-16-0.1/20251218_235027/1246"
        # }
        
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/178"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/356"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/534"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712" # Highest val
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/890"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/1068"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/1246"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/1424"
        # },
        
        # Gemma prompt 9B-IT
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/178"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/356"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/534"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/712"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/890"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1068"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1246"
        # },
        # {
        #     "adapter_path": "sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424" # Highest val
        # },
    ], 
    "llama7B": [
        #  # llama2 7B SFT - dolly prompt
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068"
        # },

        # # llama2 7B SFT - llama prompt
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068"
        # },

        # llama2 7B SFT - dolly prompt
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/534"
        # },

        # # merge all 7B dolly prompt checkpoints
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/178"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/356"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/534"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/712"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/890"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1246"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1424"
        # }

         # merge all 7B llama prompt checkpoints
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/178"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/356"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/534"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/712"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/890"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1246"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1424"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1602"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1780"
        # }
        # merge all 7B dolly prompt checkpoints
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/178"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/356"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/534"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/712"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/890"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1068"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1246"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1424"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1602"
        # },
        # {
        #     "adapter_path": "sft/7B/dolly-prompt/e32-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251218_201354/1780"
        # },
        
        
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/178"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/356"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/534"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/712"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/890"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/1068"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/1246"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/1424"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/1602"
        # },
        # {
        #     "adapter_path": "sft/7B/llama-prompt/e8-bs4-lr2e-05-G4-N4-NN1-lora-8-16-0.1/20251219_021333/1780"
        # }
    ],
    "tinyLlama1B": [
        
    #     # tinyllama 1B SFT
    #     {
    #         "adapter_path": "sft/1B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251214_233732/1068" 
    #     },

    #     # 7B (llama prompt) -> 1B FKL 0.5 - llama prompt
    #     {
    #         "adapter_path": "kd/7-1B/llama-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068/e10-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/1246"
    #     },

    #     # 7B (llama prompt) -> 1B FKL 0.5 - dolly prompt
    #     {
    #         "adapter_path": "kd/7-1B/dolly-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/llama-prompt/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068/e10-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/1068/e10-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/890"
    #     },

    #     # 7B (dolly prompt) -> 1B FKL 0.5 - dolly prompt
    #     {
    #         "adapter_path": "kd/7-1B/dolly-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251215_110135/1068",    
    #     },

    #     # 7B (dolly prompt) -> 1B FKL 1.0 - dolly prompt
    #     {
    #         "adapter_path": "kd/7-1B/dolly-prompt/1/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068/e8-bs64-lr2e-05-G1-N1-NN1-kd1.0-lora-8-16-0.1/20251215_195427/1068"
    #     },

        # 7B (dolly prompt) -> 1B FKL 0.5 - llama prompt
        # {
        #     "adapter_path": "kd/7-1B/llama-prompt/0.5/teacher-base/llama2/7b-chat-base/teacher-adapter/sft/7B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251215_013541/1068/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251218_000921/1424"
        # },

        # 7B (zeroshot) -> 1B FKL 0.5 - llama prompt
        # {
        #     "adapter_path": "kd/7-1B/llama-prompt/0.5/teacher-zeroshot/e8-bs64-lr2e-05-G1-N1-NN1-kd0.5-lora-8-16-0.1/20251218_042123/712"
        # },

    ],
    "tinyLlama1BChat": [
        #  {
        #     # "adapter_path": "sft/chat/1B/dolly-prompt/e8-bs64-lr2e-05-G1-N1-NN1-lora-8-16-0.1/20251220_185453/1068"
        #  }
     ],
    "llama3BChat":[
            # Feb 12, 2026 
            # {"adapter_path": "kd/chat/8-3B/dolly-prompt/rkl/0.5/20260209_174940/890"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/rkl/0.5/20260209_231238/890"},
            # {"adapter_path": "kd/chat/8-3B/dolly-prompt/kd/1.0/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd1.0-lora-8-16-0.1/20260211_231704/1246"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/kd/1.0/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd1.0-lora-8-16-0.1/20260212_042850/1424"}
            
            
            
            # # #  Dolly prompt
            # {"adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246"},
            # # # Llama prompt
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424"},
            
            # For FKL+RKL
            # {"adapter_path": "kd/chat/8-3B/dolly-prompt/fkl+rkl/0.5/20260218_060855/1246"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/fkl+rkl/0.5/20260218_012550/712"},
            
            # # SFT dolly prompt
            # {"adapter_path": "sft/llama3b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_170116/890"},
            # SFT llama prompt
            # {"adapter_path": "sft/llama3b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251225_200923/1246"},
            
            # {"adapter_path": "kd/chat/8-3B/teacher-chat-data-dir/student-nonchat-data-dir/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260101_224450/890"},
            # {"adapter_path": "kd/chat/8-3B/teacher-chat-data-dir/student-nonchat-data-dir/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt-test/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_021124/890"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260103_120455/534"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_220635/890"},
            # {"adapter_path": "kd/chat/8-3B/llama-prompt/mixing/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e6-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260102_160837/890"},
            
            
            # ALL ckpt for dolly prompt
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/178"
            # },
            
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/356"
            # },
            
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/534"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/712"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/890"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1068"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1424"
            # },
            
            # ALL ckpt for llama prompt
            
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/178"
            # },
            
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/356"
            # },
            
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/534"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/712"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/890"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1068"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1246"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424"
            # },
            # Mixing prompt
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_022553/712"
            # },
            # {
            #    "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_115557/1246" 
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260215_191116/890"
            # },
            # crisscross evaluation
            # {
            #     "adapter_path": "kd/chat/8-3B/dolly-prompt/teacher-criscross-evaluation/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260406_223602/712"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/teacher-criscross-evaluation/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260407_143936/534"
            # }
            
            # Prompt mixing 
            
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.1/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260416_124134/890"
            # },
            # {
            #    "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_115557/1246" 
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.4/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260418_022406/890"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260215_191116/890"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/curriculum_0.6/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260418_102456/1424"
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260216_022553/712"
            # }
            ##### MEDQA CKPTS.
            # {
            #     "adapter_path": "sft/llama3b-it/med-qa/llama-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05/20260512_221743/312" # 3B CKPT SFT FOR LLAMA Prompt.
            # },
            # {
            #     "adapter_path": "sft/llama3b-it/med-qa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05/20260512_003339/624" # 3B CKPT SFT FOR medqa Prompt.
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/llama-prompt/med-qa/0.5/teacher-base/llama-8B/teacher-adapter/sft_llama8b-it_med-qa_llama-prompt_e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05_20260510_103454_624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260510_220342/624" # KD ckpts with llamaprompt.
            # },
            # {
            #     "adapter_path": "kd/chat/8-3B/medqa-prompt/med-qa/0.5/teacher-base/llama-8B/teacher-adapter/sft_llama8b-it_med-qa_medqa-prompt_e3-bs2-lr0.0001-G16-N4-NN1-lora-16-32-0.05_20260511_192134_312/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260512_172436/312" # KD ckpts with medqaprompt.
            # },
            # {
            #     "adapter_path": "sft/llama3b-it/med-qa/llama-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.05/20260515_194048/936" # 3B CKPT SFT FOR LLAMA Prompt. r=8 and 16
            # },
            # {
            #     "adapter_path": "sft/llama3b-it/med-qa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.05/20260515_221520/624" # 3B CKPT SFT FOR medqa Prompt. r=8 and 16
            # }
            
            
        ],
    "llama8BChat":[
            # {'adapter_path': "sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068"},
            # {'adapter_path': "sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068"}  
        ],
    "qwen3BChat": [
        # {'adapter_path': "sft/qwen2.5-3B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_032105/1424"},
        # {'adapter_path': "sft/qwen2.5-3B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_112846/1068"},
        # {'adapter_path': "kd/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_085107/1424"},
        # {'adapter_path': "kd/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_024037/1424"},
        # {'adapter_path': 'sft/qwen2.5-3B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_021754/624'},
        # {'adapter_path': 'sft/qwen2.5-3B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260522_053035/936'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/936'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/624'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/312'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/624'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/medqa-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260522_181643/936'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/312'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/624'},
        # {'adapter_path': 'kd/chat/7-3B/medqa/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936/e3-bs1-lr0.0001-G32-N4-NN1-kd0.5-lora-8-16-0.1/20260523_011933/936'},
        {'adapter_path': 'rkl/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260524_133641/1068'},
        {'adapter_path': 'rkl/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260523_194624/534'}
        
        
        ],
    "qwen7BChat": [
        # {'adapter_path': "sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068"},
        # {'adapter_path': "sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246"},
        # {'adapter_path': 'sft/qwen2.5-7B-it/medqa/qwen-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_213309/936'},
        # {'adapter_path': 'sft/qwen2.5-7B-it/medqa/medqa-prompt/e3-bs2-lr0.0001-G16-N4-NN1-lora-8-16-0.1/20260521_134300/624'}
    ]


}

for model in models:
    
    if model == 'gemma2BChat':
        base_model_path="/media/drive2/models/gemma-models/gemma-2-2b-it" 
        adapter_path_prefix=f"{DRIVE_PATH}/results/gemma2"

    elif model == 'gemma9BChat':
        base_model_path="/media/drive2/models/gemma-models/gemma-2-9b-it"
        adapter_path_prefix=f"{DRIVE_PATH}/results/gemma2"

    elif model == 'tinyLlama1B':
        base_model_path=f"{DRIVE_PATH}/checkpoints/llama2/TinyLlama_v1.1"
        adapter_path_prefix=f"{DRIVE_PATH}/results/llama2"
    elif model == 'tinyLlama1BChat':
        base_model_path=f"{DRIVE_PATH}/checkpoints/llama2/TinyLlama-1.1B-Chat-v1.0"
        adapter_path_prefix=f"{DRIVE_PATH}/results/llama2"

    elif model == 'llama7B':
        base_model_path="/media/drive2/models/llama-models/Llama-2-7b-chat-hf"
        adapter_path_prefix=f"{DRIVE_PATH}/results/llama2"
    elif model == 'llama3BChat':
        base_model_path="/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
        adapter_path_prefix=f"{DRIVE_PATH}/results/llama3"
    elif model == 'llama8BChat':
        base_model_path="/media/drive2/models/llama-models/Llama-3.1-8B-Instruct"
        adapter_path_prefix=f"{DRIVE_PATH}/results/llama3"
        
    elif model == 'qwen3BChat':
        base_model_path="/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct"
        adapter_path_prefix=f"{DRIVE_PATH}/results/qwen2"
    elif model == 'qwen7BChat':
        base_model_path="/media/drive2/models/qwen-models/Qwen2.5-7B-Instruct"
        adapter_path_prefix=f"{DRIVE_PATH}/results/qwen2"
    else:
        raise RuntimeError("Please provide base model path")        
        
    for adapter in models[model]:
        adapter_path = adapter_path_prefix + '/train/' + adapter["adapter_path"]
        merged_path  = adapter_path_prefix + '/merged/' + adapter["adapter_path"]
        print(f"Merging base model: {base_model_path} with adapter: {adapter_path} into {merged_path}")
        merge_model(base_model_path, adapter_path, merged_path)
