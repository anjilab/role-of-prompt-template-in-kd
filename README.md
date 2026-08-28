# Understanding the Role of Prompt Template in Knowledge Distillation for Safety Alignment

This repository contains training, distillation, utility-evaluation, safety-evaluation, and analysis code for studying how knowledge distillation and prompt formats affect utility and safety behavior in instruction-tuned LLMs.

> **NOTE:** Most of the training and evaluation infrastructure in this codebase is adopted from [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) (Sun et al., 2024). We extend it with safety evaluation pipelines, prompt-format variants (chat vs. non-chat KD), criss-cross and curriculum mixing strategies, and mechanistic analysis tooling.

The code supports experiments across Qwen2.5, Llama 3/3.2, and Gemma 2 model families. 

## Repository Layout

```text
.
├── analysis/                         # Representation and refusal-direction analyses
├── configs/
│   ├── deepspeed/                    
│   └── hostfiles/                    
├── data/                             
├── data_utils/                       # Dataset loading and indexed dataset utilities
├── processed_data/                   # Tokenized data used by training/evaluation scripts
├── scripts/
│   ├── gemma/                        # Gemma SFT, KD, evaluation, and preprocessing scripts
│   ├── llama3/                       # Llama SFT, KD, evaluation, and preprocessing scripts
│   └── qwen2/                        # Qwen SFT, KD, evaluation, and preprocessing scripts
├── tools/                            # Data processing and checkpoint conversion tools
├── walledeval/                       # Safety evaluation with WalledEval + LlamaGuard
├── finetune.py                       # Entry point
├── finetune_bits_and_bytes.py        
├── evaluate.py                       
├── evaluate_main.py                  
├── generate.py                       
├── model_evaluation_configs.yaml     # Model registry for utility evaluation
└── run_model_utility_evaluation.sh   # Utility evaluation launcher
```

## Setup

Create and activate a Python environment, then install the project dependencies:

```bash
bash install.sh
```

`install.sh` installs the MiniLLM-compatible Transformers fork, PyTorch CUDA wheels, DeepSpeed, PEFT, datasets, bitsandbytes, and evaluation dependencies.

If you install dependencies manually, the core packages are:

```text
torch
deepspeed
transformers from git+https://github.com/t1101675/transformers@minillm
accelerate
datasets
peft
bitsandbytes
sentencepiece
protobuf
rouge-score
torchtyping
numerize
rich
wandb
```


## Data

The repository includes raw and processed data folders used by the experiments:

- `data/dolly`: Dolly instruction data used for training.
- `data/vicuna`: Vicuna evaluation prompts.
- `data/sinst`: Super-Natural-Instructions evaluation prompts.
- `data/harm_bench`: HarmBench test cases and saved evaluation results.
- `data/walledai`: WalledAI safety datasets such as AdvBench, JailbreakBench, and XSTest.
- `processed_data`: tokenized datasets used by training and evaluation scripts.

To process Dolly-style data for Qwen, run:

```bash
bash scripts/qwen2/tools/process_data_dolly.sh
```

To run preprocessing scripts:

```bash
bash scripts/llama3/tools/process_data_dolly.sh
```

## Training Models

Training is organized by model family and method:

- `scripts/*/sft/`: supervised fine-tuning scripts.
- `scripts/*/kd/`: knowledge distillation scripts.
- `scripts/*/kd/task_prompt/`: KD with task/non-chat prompts.
- `scripts/*/kd/chat_prompt/`: KD with chat prompts.
- `scripts/*/kd/prompt_mix/`: prompt-mixing/curriculum KD variants.

# Training methods

## SFT
```bash
bash scripts/qwen2/sft/task_prompt/sft_3B_lora_dolly.sh
```
## KD

```bash
bash scripts/qwen2/kd/task_prompt/kd_3B_7B_teacher_dolly_lora_dolly.sh
```


## Utility Evaluation

Utility evaluation is driven by `model_evaluation_configs.yaml` and `run_model_utility_evaluation.sh`.

1. Register the model to evaluate in `model_evaluation_configs.yaml`.
2. Ensure the config entry includes:
   - `ckpt_name`
   - `ckpt_path`
   - `peft_ckpt_name`
   - `peft_ckpt_path`
   - `CUDA_DEVICES`
   - `gpus_per_node`
   - `save_root`
   - `eval_method`
   - `eval_family`
   - `eval_technique`
   - `prompt_type`
   - `eval_script`
3. Run one or more registered configs:

```bash
bash run_model_utility_evaluation.sh qwen_2_3b_it_rkl_qwen_prompt_for_dolly
```

If no config key is passed, the launcher uses the default `MODEL_CONFIGS` list inside `run_model_utility_evaluation.sh`.

The utility evaluation scripts evaluate instruction-following datasets such as Dolly, Self-Instruct, Vicuna, and Super-Natural-Instructions. `evaluate_main.py` writes model predictions, answers, logs, and metrics to the configured save directory.

## Safety Evaluation

For safety evaluation: 
# 1. Clone HarmBench next to this repo
- `git clone https://github.com/centerforaisafety/HarmBench $BASE_PATH/HarmBench`
- `cd $BASE_PATH/HarmBench`
- `pip install -r requirements.txt`

# 2 HarmBench test behaviors CSV at data/behavior_datasets/

# 3. Place or generate test cases at:
#    $BASE_PATH/data/harm_bench/test_cases/test_cases.json
#    (HarmBench's attack scripts generate this; or use the pre-generated ones from the repo)


For <b> SORRYBench: </b>
# 1. Clone SorryBench next to this repo
- `git clone https://github.com/sorry-bench/sorry-bench $BASE_PATH/sorry-bench`
- `cd $BASE_PATH/sorry-bench`
- `pip install -r requirements.txt`

# 2. Data are at:  $BASE_PATH/data/sorry_bench/question.jsonl

After setup you can run:

- `bash tools/eval_harmbench.sh`
- `bash tools/run_sorry.sh`
- `bash tools/run_parallel_eval.sh` This one is for walledeval. 


## Analysis

Analysis scripts are in `analysis/`. The current revised scripts include:

- `analysis/refusal_direction_revised.py`
- `analysis/pca_layerwise_all_revised.py`

Generated figures and summaries are stored under `analysis/outputs/`, including PCA layerwise plots and refusal-direction metrics.

See `analysis/README.md` for a detailed analysis guide. Some script names in that guide refer to earlier prototype filenames, so check the actual files in `analysis/` before running.

