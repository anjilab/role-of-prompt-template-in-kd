# Analysis Folder Guide

This folder contains mechanistic and behavior analyses for comparing:
- `Base`
- `KD-Chat`
- `KD-NonChat`

Most scripts focus on safety-relevant representation/attention changes across layers.

## Important Notes Before Running

- Many scripts use hardcoded model/adaptor paths (for example under `/media/drive2/...`).
- Several scripts load:
  - `data/sorry_bench/question.jsonl` (harmful prompts)
  - `databricks/databricks-dolly-15k` from Hugging Face (harmless prompts)
- Set paths inside each script before execution.
- Default device is CUDA when available.

## Main Scripts

1. `refusal_direction_revised.py`
- Purpose: Layerwise refusal-direction analysis (Base vs KD variants), cosine similarities, dashboards.
- Output: `analysis/outputs/refusal_direction_revised/`
- Run:
```bash
python analysis/refusal_direction_revised.py
```



2. `pca_layerwise_all_revised.py`
- Purpose: PCA visualization of harmful vs harmless residual activations across layers/models.
- Output: `analysis/outputs/pca_layerwise_all_revised/`
- Run:
```bash
python analysis/pca_layerwise_all_revised.py
```

