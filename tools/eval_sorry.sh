BASE_PATH="/home/minillm_ai_safety"

# Declare an array of "records"
declare -a models

# Single array: each entry = "relative_path===model_name"
models=(
    # "data/sorry_bench/model_judgment/llama2/base-llama7B-adapter-llama-2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-llama7B-adapter-llama-2"
    # "data/sorry_bench/model_judgment/llama2/base-tinyllama1B-adapter-dolly-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-tinyllama1B-adapter-dolly-v2"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-prompt"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-llama-prompt"
    # "data/sorry_bench/model_judgment/llama2/sft-tinyllama1B-adapter-dolly-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-tinyllama1B-adapter-dolly-v2"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-dolly-prompt"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-llama7B-llama-prompt-tinyllama1B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-llama7B-llama-prompt-tinyllama1B-dolly-prompt"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-llama7B-llama-prompt-tinyllama1B-llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-llama7B-llama-prompt-tinyllama1B-llama-prompt"
    # "data/sorry_bench/model_judgment/llama2/kd-1.0-llama7B-dolly-prompt-tinyllama1B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-1.0-llama7B-dolly-prompt-tinyllama1B-dolly-prompt"
    # "data/sorry_bench/model_judgment/llama2/base-llama7B/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-llama7B"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-tiny_llama_1B_0_5_0_seed_42_fkl/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-tiny_llama_1B_0_5_0_seed_42_fkl"
    # "data/sorry_bench/model_judgment/llama2/kd-1.0-tiny_llama_1B_1_0_seed_42_fkl/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-1.0-tiny_llama_1B_1_0_seed_42_fkl"
    # "data/sorry_bench/model_judgment/llama2/sft-tinyllama1B/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-tinyllama1B"
    # "data/sorry_bench/model_judgment/llama2/tiny_llama_1B_0_5_seed_42_fkl_llama_prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/tiny_llama_1B_0_5_seed_42_fkl_llama_prompt"

    # "data/sorry_bench/model_judgment/kc-academy/llama-2-1B-kd0.5-LTLP/ft-mistral-7b-instruct-v0.2.jsonl===kc-academy/llama-2-1B-kd0.5-LTLP"
    # "data/sorry_bench/model_judgment/kc-academy/llama-2-1B-kd0.5-DTLP/ft-mistral-7b-instruct-v0.2.jsonl===kc-academy/llama-2-1B-kd0.5-DTLP"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/base-llama7B_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/base-llama7B_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/base-llama7B_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/base-llama7B_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/base-tinyllama1B_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/base-tinyllama1B_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/base-tinyllama1B_llama2adapte/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/base-tinyllama1B_llama2adapte"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/dolly-prompt/1424/dolly_prompt_sft_7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/llama-prompt/890/llama_prompt_sft_7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/dolly-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/dolly-prompt/7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/kd/7-1B/llama-prompt/0.5/teacher/7B/llama-prompt/7b_llama2adapter"


    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-prompt-with-dolly-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-prompt-with-dolly-adapter"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-prompt-with-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-prompt-with-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-llama-prompt-with-dolly-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-llama-prompt-with-dolly-adapter"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-llama-prompt-with-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-llama-prompt-with-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-llama7B-zeroshot-tinyllama1B-llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-llama7B-zeroshot-tinyllama1B-llama-prompt"
    # "data/sorry_bench/model_judgment/llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/kd-0.5-llama7B-dolly-prompt-tinyllama1B-llama-prompt"
    
    #try
    # "data/sorry_bench/model_judgment/llama2/base-tinyllama1B-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-tinyllama1B-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/sft-tinyllama1B-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-tinyllama1B-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-prompt-with-llama-adapter-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-prompt-with-llama-adapter-v2"
    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-llama-prompt-with-llama-adapter-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-llama-prompt-with-llama-adapter-v2"

    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-534ckpt-prompt-llama-adapter-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-534ckpt-prompt-llama-adapter-v2"

    # "data/sorry_bench/model_judgment/llama2/sft-llama7B-dolly-178ckpt-prompt-llama-adapter-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/sft-llama7B-dolly-178ckpt-prompt-llama-adapter-v2"
  
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_dollyv2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_dollyv2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/1B/dolly-prompt/1424/dolly_prompt_sft_1b_llama2adapter"


    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/llama-prompt/890/verify_llama_prompt_sft_7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/llama-prompt/890/verify_llama_prompt_sft_7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/dolly-prompt/1424/verify_dolly_prompt_sft_7b_llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/dolly-prompt/1424/verify_dolly_prompt_sft_7b_llama2adapter"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/llama-prompt/kc-academy/luffy-sft-llama-7B-LP/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/llama-prompt/kc-academy/luffy-sft-llama-7B-LP"
    # "data/sorry_bench/model_judgment/llama2/reproducibility/sft/7B/dolly-prompt/1068-ckpt-llama2adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/reproducibility/sft/7B/dolly-prompt/1068-ckpt-llama2adapter"



    # "data/sorry_bench/model_judgment/llama2/v2/base-llama7B/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/base-llama7B"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-178-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-178-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-356-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-356-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-534-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-534-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-712-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-712-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-890-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-890-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-1068-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-1068-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-1246-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-1246-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-1424-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-1424-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-1602-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-1602-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v2/sft-llama7B-dolly-1780-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v2/sft-llama7B-dolly-1780-llama-adapter"


    # "data/sorry_bench/model_judgment/llama2/v3/base-llama7B/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/base-llama7B"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-178-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-178-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-356-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-356-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-534-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-534-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-712-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-712-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-890-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-890-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-1068-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-1068-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-1246-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-1246-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-1424-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-1424-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-1602-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-1602-llama-adapter"
    # "data/sorry_bench/model_judgment/llama2/v3/sft-llama7B-llama-1780-llama-adapter/ft-mistral-7b-instruct-v0.2.jsonl===llama2/v3/sft-llama7B-llama-1780-llama-adapter"

    # "data/sorry_bench/model_judgment/llama2/base-tinyllama1B-adapter-dolly-v2/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-tinyllama1B-adapter-dolly-v2"
    # "data/sorry_bench/model_judgment/llama2/base-tinyllama1.1B-chat/ft-mistral-7b-instruct-v0.2.jsonl===llama2/base-tinyllama1.1B-chat"

    # "data/sorry_bench/model_judgment/llama2/chat/base-tinyllama1.1B-chat/ft-mistral-7b-instruct-v0.2.jsonl===llama2/chat/base-tinyllama1.1B-chat"
    # "data/sorry_bench/model_judgment/llama2/chat/sft-tinyllama1B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama2/chat/sft-tinyllama1B-dolly-prompt"

    # "data/sorry_bench/model_judgment/gemma2/base-gemma2B/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/base-gemma2B"
    # "data/sorry_bench/model_judgment/gemma2/sft-gemma2B-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/sft-gemma2B-dolly-prompt"

    # "data/sorry_bench/model_judgment/llama2/chat/tinyllama-base/ft-mistral-7b-instruct-v0.2.jsonl===llama2/chat/tinyllama-base"

    # "data/sorry_bench/model_judgment/llama2/hypothesis/teacher_base/ft-mistral-7b-instruct-v0.2.jsonl===llama2/hypothesis/teacher_base"
    # "data/sorry_bench/model_judgment/llama2/hypothesis/teacher_sft/ft-mistral-7b-instruct-v0.2.jsonl===llama2/hypothesis/teacher_sft"
    # "data/sorry_bench/model_judgment/llama2/hypothesis/rkl/ft-mistral-7b-instruct-v0.2.jsonl===llama2/hypothesis/rkl"
    # "data/sorry_bench/model_judgment/llama2/hypothesis/fkl/ft-mistral-7b-instruct-v0.2.jsonl===llama2/hypothesis/fkl"

    # # gemma 2B
    # "data/sorry_bench/model_judgment/gemma2/chat/base-gemma2B-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/base-gemma2B-gaamma-adapter"

    # #  # All gemma 2B checkpoints - dolly prompt

    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-dolly-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-dolly-1424-gaamma-adapter"
    # #  # All gemma 2B checkpoints - gemma prompt
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/sft-gemma2B-gemma-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/sft-gemma2B-gemma-1424-gaamma-adapter"
    

    ## All gemma 9b it ckpts - dolly prompt

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/base-gemma2B-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/base-gemma2B-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/base-gemma9B-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/base-gemma9B-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft-gemma9B-gemma-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft-gemma9B-gemma-1424-gaamma-adapter"

    # # All gemma 2B checkpoints - dolly prompt - 1e-5
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-5/sft-gemma2B-dolly-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-5/sft-gemma2B-dolly-1424-gaamma-adapter"

    # # All gemma 2B checkpoints - dolly prompt - 3e-5
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/3e-5/sft-gemma2B-dolly-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/3e-5/sft-gemma2B-dolly-1424-gaamma-adapter"
    
    # # # All gemma 2B checkpoints - dolly prompt - 2e-4
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/2e-4/sft-gemma2B-dolly-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/2e-4/sft-gemma2B-dolly-1424-gaamma-adapter"

    # # # All gemma 2B checkpoints - dolly prompt - 1e-4
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-178-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-178-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-356-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-356-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-534-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-534-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-712-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-712-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-890-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-890-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-1068-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-1068-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-1246-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-1246-gaamma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/1e-4/sft-gemma2B-dolly-1424-gaamma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/1e-4/sft-gemma2B-dolly-1424-gaamma-adapter"


    ## All gemma 9B ckpts - gemma prompt
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-178-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-178-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-534-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-534-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-712-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-712-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-890-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-890-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1068-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1068-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1246-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1246-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1424-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9B/gemma_prompt/sft-gemma9B-gemma-1424-gemma-adapter"

    ## All gemma 9B-2B KD ckpts - kd = dolly prompt and teacher = dolly
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter"


    #  ## All gemma 9B-2B KD ckpts - kd = gemma prompt and teacher = gemma
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter"


    ###### LLama3.2 version
    
    # running all gemma KD 9B-2B IT - 0.5- gemma prompt
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-178-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-178-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-356-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-356-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-534-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-534-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-712-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-712-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-890-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-890-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-1068-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-1068-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-1246-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-1246-gemma-adapter"
    # "data/sorry_bench/model_judgment/gemma2/chat/kd/0.5/9-2B-gemma-1424-gemma-adapter/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/kd/0.5/9-2B-gemma-1424-gemma-adapter"


    # # llama_3_3B Base Model
    # "data/sorry_bench/model_judgment/llama3/base-3B/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B"

    # # llama_3_8B Base Model
    # "data/sorry_bench/model_judgment/llama3/base-8B/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-8B"



    # # llama_3_3B Base Model
    # "data/sorry_bench/model_judgment/llama3/chat/base/3B/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/base/3B"
    # # llama_3_8B Base Model
    # "data/sorry_bench/model_judgment/llama3/chat/base/8B/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/base/8B"

    # #llama_3_8B_dollyprompt sft
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/dolly-prompt"
    # #llama_3_8B_llamaprompt sft
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/llama-prompt"

    #llama_3_3B_dollyprompt sft
    # "data/sorry_bench/model_judgment/llama3/chat/sft/3B/dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/3B/dolly-prompt"
    # #llama_3_3B_llamaprompt sft
    # "data/sorry_bench/model_judgment/llama3/chat/sft/3B/llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/3B/llama-prompt"

    # #llama_3_8_3B kd = dolly promot teacher = dolly prompt
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt"
    # #llama_3_8_3B  kd = llama promot teacher = llama prompt
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt"


    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50"

    # "data/sorry_bench/model_judgment/llama3/base-3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed10"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed20"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed30"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed40"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed50"

    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed50"



    ### Gemma
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base-gemma2B-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base-gemma2B-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base-gemma2B-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base-gemma2B-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base-gemma2B-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base-gemma2B-gemma-adapter/seed30"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base-gemma2B-gemma-adapter/seed40/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base-gemma2B-gemma-adapter/seed40"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base-gemma2B-gemma-adapter/seed50/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base-gemma2B-gemma-adapter/seed50"

    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50"

    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed30"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed40"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/kd/gemma_prompt/3B/teacher-gemma-prompt/seed50"



    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed30"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed40/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed40"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed50/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_gemma_prompt_1424_9b-it/seed50"

    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed30"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed40/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed40"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed50/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/sft_dolly_prompt_712_9b-it/seed50"

    ## LLama Models

    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/dolly_prompt/3B/teacher-dolly-prompt/seed50"

    # "data/sorry_bench/model_judgment/llama3/base-3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed10"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed20"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed30"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed40"
    # "data/sorry_bench/model_judgment/llama3/base-3B/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/base-3B/seed50"

    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/kd/0.5/llama_prompt/3B/teacher-llama-prompt/seed50"

    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_dolly_prompt_1068_8b-it/seed50"

    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed30"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed40/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed40"
    # "data/sorry_bench/model_judgment/llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed50/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed_chat/sft/sft_llama_prompt_1068_8b-it/seed50"




    # All gemma 9B-2B KD ckpts - kd = dolly prompt and teacher = dolly

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/dolly_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed30"


    # All gemma 9B-2B KD ckpts - kd = gemma prompt and teacher = gemma
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-178-gemma-adapter/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-356-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-534-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-712-gemma-adapter/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-890-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1068-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1246-gemma-adapter/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/kd/gemma_prompt/kd-gemma9B-gemma-1424-gemma-adapter/seed30"





    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-178-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-356-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-534-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-712-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-890-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1068-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1246-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/dolly_prompt/kd-llama8B-llama-1424-llama-adapter/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-178-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-356-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-534-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-712-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-890-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1068-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1246-llama-adapter/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/kd/llama_prompt/kd-llama8B-llama-1424-llama-adapter/seed30"


    ### 
    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed10"
    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed20"
    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed30"

    # "data/sorry_bench/model_judgment/llama3/metamath/base/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/3B/seed10"
    # "data/sorry_bench/model_judgment/llama3/metamath/base/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/3B/seed20"
    # "data/sorry_bench/model_judgment/llama3/metamath/base/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/3B/seed30"


    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/dolly-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/rkl/llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/rkl/llama-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/dolly-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/v2/3B/llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/v2/3B/llama-prompt/seed30"


    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/dolly-prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/rkl/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/rkl/gemma-prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/dolly_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/dolly_prompt//seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/dolly_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/dolly_prompt//seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/dolly_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/dolly_prompt//seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/gemma_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/gemma_prompt//seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/gemma_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/gemma_prompt//seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/sft/2B/gemma_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/sft/2B/gemma_prompt//seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.2/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.5/seed30"
    
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-llama8B-llama-mixing-0.8/seed30"

    # "data/sorry_bench/model_judgment/llama3/seed/v2/latest/base-8B-it/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed/v2/latest/base-8B-it/seed10"
    # "data/sorry_bench/model_judgment/llama3/seed/v2/latest/base-8B-it/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed/v2/latest/base-8B-it/seed20"
    # "data/sorry_bench/model_judgment/llama3/seed/v2/latest/base-8B-it/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/seed/v2/latest/base-8B-it/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/llama-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/sft/8B/latest/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/sft/8B/latest/dolly-prompt/seed30"
    

    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/dolly-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/kd/0.5/8-3B/teacher-crisscross-eval/llama-prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.1/llama_prompt/seed30"


    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.2/llama_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.4/llama_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.5/llama_prompt/seed30"


    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.6/llama_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/8-3B/mixing/kd-0.5/curriculum/0.8/llama_prompt/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/dolly-prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/teacher-crisscross-eval/gemma-prompt/seed30"


    # Mixing one.

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.4/gemma-prompt/seed30"



    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.5/gemma-prompt/seed30"



    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.6/gemma-prompt/seed30"


    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/9-2B/mixing/curriculum_0.8/gemma-prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed10"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed20"
    # "data/sorry_bench/model_judgment/gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/seed/v2/base/gemma-9B-gemma-adapter/seed30"


    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed10",
    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed20",
    # "data/sorry_bench/model_judgment/llama3/metamath/base/1B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/metamath/base/1B/seed30",


    # "data/sorry_bench/model_judgment/llama3/chat/base/1B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/base/1B/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/base/1B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/base/1B/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/base/1B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/base/1B/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/llama-prompt/sft_3b_it_medqa_llama_prompt_r=8-16/seed30"


    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/3B/sft/medqa-prompt/sft_3b_it_medqa_medqa_prompt_r=8-16/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/llama-prompt/kd_0.5_llama_prompt/seed30"

    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed10"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed20"
    # "data/sorry_bench/model_judgment/llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===llama3/chat/v2/medqa/8-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/gemma-prompt/kd_0.5_gemma_prompt/seed30"

    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed10"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed20"
    # "data/sorry_bench/model_judgment/gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===gemma2/chat/v2/medqa/9-2B/kd/medqa-prompt/kd_0.5_medqa_prompt/seed30"

    # "data/sorry_bench/model_judgment/qwen2/base/1B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/1B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/base/1B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/1B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/base/1B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/1B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/base/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/3B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/base/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/3B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/base/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/3B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/base/7B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/7B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/base/7B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/7B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/base/7B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/base/7B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/3B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/dolly-prompt/7B/seed30"
    

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/3B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/dolly/qwen-prompt/7B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/dolly/kd_0.5_dolly_prompt/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/kd/dolly/kd_0.5_qwen_prompt/seed30"

    ### MEDQA
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/3B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/3B/seed30"


    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/medqa-prompt/7B/seed30"

    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed10"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed20"
    # "data/sorry_bench/model_judgment/qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/seed/v2/sft/medqa/qwen-prompt/7B/seed30"
    
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_medqa_prompt/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa/kd_0.5_qwen_prompt/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt1/seed30"


    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt2/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_qwen_prompt_ckpt3/seed30"
    

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt1/seed30"


    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt2/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/kd/7-3B/medqa-v2/kd_0.5_medqa_prompt_ckpt3/seed30"
    "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed10"
    "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed20"
    "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_dolly_prompt/seed30"

    # "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed10/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed10"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed20/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed20"
    # "data/sorry_bench/model_judgment/qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed30/ft-mistral-7b-instruct-v0.2.jsonl===qwen2/chat/v2/rkl/7-3B/dolly/kd_0.5_qwen_prompt/seed30"
)      



for entry in "${models[@]}"; do
    # Split into path and name
    rel_path="${entry%%===*}"
    model_name="${entry#*===}"

    full_path="${BASE_PATH}/${rel_path}"

    echo "=================================================="
    echo "Model Name : $model_name"
    echo "Data Path  : $rel_path"
    echo "Full Path  : $full_path"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=5 python "${BASE_PATH}/sorry-bench/visualize_result.py" \
        --data-path "$full_path" \
        --model-name "$model_name"

    if [[ $? -ne 0 ]]; then
        echo "Error: Failed on $model_name"
    fi

    echo
done

echo "All done!"