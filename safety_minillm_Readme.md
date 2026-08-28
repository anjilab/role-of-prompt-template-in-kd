# MiniLLM AI Safety README

The main repository README is available in [`README.md`](README.md). It covers:

- setup and dependencies
- repository layout
- data processing
- SFT and KD training scripts
- utility evaluation through `model_evaluation_configs.yaml`
- safety evaluation through `walledeval/eval_safety.py`
- analysis scripts and output locations

Safety note: do not use Qwen-2.5-Math models for safety evaluation. The model series is not recommended for non-math tasks, so it is not appropriate for the safety experiments in this repository.
