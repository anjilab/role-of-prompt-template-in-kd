import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import csv
import torch
import functools
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
from peft import PeftModel

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

_MIX_PREFIX = "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5"
_MIX_SUFFIX = "teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1"

LLAMA_CONFIGS = {
    "Base":         "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct",
    "KD-Chat":      "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424",
    "KD-NonChat":   "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246",
    "KD-Mix":       f"{_MIX_PREFIX}/{_MIX_SUFFIX}/20260216_115557/1246",
    "KD-Mix-C0.6":  f"{_MIX_PREFIX}/curriculum_0.6/{_MIX_SUFFIX}/20260418_102456/1424",
}

GEMMA_CONFIGS = {
    "Base":       "/media/drive2/models/gemma-models/gemma-2-2b-it",
    "KD-Chat":    "/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246",
    "KD-NonChat": "/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424",
}

QWEN_CONFIGS = {
    "Base":       "/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct",
    "KD-Chat":    "/media/drive2/distillation/safety-utility-pg/results/qwen2/train/kd/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_024037/1424",
    "KD-NonChat": "/media/drive2/distillation/safety-utility-pg/results/qwen2/train/kd/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_085107/1424",
}

LLAMA_TL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA_TL_NAME = "google/gemma-2-2b-it"
QWEN_TL_NAME = "Qwen/Qwen2.5-3B-Instruct"


LLAMA_CHAT_TEMPLATE = None
GEMMA_CHAT_TEMPLATE = None
QWEN_CHAT_TEMPLATE = None


N_INST_TRAIN = 32

OUTPUT_DIR = Path(__file__).parent / "outputs" / "refusal_direction_revised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def patch_qwen_rope_theta_for_transformer_lens():
    original_from_pretrained = AutoConfig.from_pretrained

    @functools.wraps(original_from_pretrained)
    def from_pretrained_with_qwen_rope_theta(*args, **kwargs):
        config = original_from_pretrained(*args, **kwargs)
        rope_parameters = getattr(config, "rope_parameters", None)
        if (
            getattr(config, "model_type", None) == "qwen2"
            and not hasattr(config, "rope_theta")
            and isinstance(rope_parameters, dict)
            and "rope_theta" in rope_parameters
        ):
            config.rope_theta = rope_parameters["rope_theta"]
        return config

    AutoConfig.from_pretrained = from_pretrained_with_qwen_rope_theta


patch_qwen_rope_theta_for_transformer_lens()

# ── Visual style per model ──
STYLE = {
    "Base":         dict(color="#555555", linestyle="--", linewidth=1.5),
    "KD-Chat":      dict(color="#E07B39", linestyle="-",  linewidth=2.0),
    "KD-NonChat":   dict(color="#3A7EBF", linestyle="-",  linewidth=2.0),
    "KD-Mix":       dict(color="#2CA02C", linestyle="-",  linewidth=1.8),
    "KD-Mix-C0.6":  dict(color="#9467BD", linestyle="-",  linewidth=1.8),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def load_questions(question_file: str, begin: int, end: int):
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line:
                questions.append(json.loads(line))
    return questions[begin:end]


def get_harmful_instructions():
    question_file = "data/sorry_bench/question.jsonl"
    harmful_questions = load_questions(question_file, begin=0, end=200)
    instructions = [q['turns'][0] for q in harmful_questions]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    instructions = [row['instruction'] for row in dataset.select(range(200))]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    template: str | None,
) -> Int[Tensor, "batch seq"]:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if template is not None:
        kwargs["chat_template"] = template
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": i}], **kwargs)
        for i in instructions
    ]
    return tokenizer(
        prompts, padding=True, truncation=False, return_tensors="pt"
    ).input_ids


# ═══════════════════════════════════════════════════════════════════════════════
# REFUSAL DIRECTION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_refusal_directions_all_layers(
    model: HookedTransformer,
    harmful_toks: Tensor,
    harmless_toks: Tensor,
    pos: int = -1,
    chunk_size: int = 4,
) -> Float[Tensor, "n_layers d_model"]:
    """
    r_l^M = normalize( E[h_l^M(x) | x in D_harm] - E[h_l^M(x) | x in D_safe] )
    Computed at every layer using resid_pre activations at position `pos`.
    """
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    device   = next(model.parameters()).device

    def mean_acts(toks: Tensor, label: str) -> Tensor:
        acc = torch.zeros(n_layers, d_model, device=device, dtype=torch.float32)
        n   = toks.shape[0]
        print(f"    [{label}] processing {n} samples (chunk={chunk_size})...")
        for i in range(0, n, chunk_size):
            chunk = toks[i : i + chunk_size]
            _, cache = model.run_with_cache(
                chunk, names_filter=lambda name: "resid_pre" in name
            )
            for layer in range(n_layers):
                acc[layer] += cache["resid_pre", layer][:, pos, :].float().sum(dim=0)
            del cache
            gc.collect()
            torch.cuda.empty_cache()
        return acc / n

    harm_acts     = mean_acts(harmful_toks,  "harmful")
    harmless_acts = mean_acts(harmless_toks, "harmless")

    directions = []
    for layer in range(n_layers):
        d = harm_acts[layer] - harmless_acts[layer]
        directions.append(d / (d.norm() + 1e-8))

    del harm_acts, harmless_acts
    gc.collect()
    torch.cuda.empty_cache()

    return torch.stack(directions, dim=0)   # (n_layers, d_model)


def cosine_sim_per_layer(
    dirs_a: Float[Tensor, "L D"],
    dirs_b: Float[Tensor, "L D"],
) -> np.ndarray:
    """CosSim_l(M, M_S) — element-wise, both inputs already unit-norm."""
    sims = (dirs_a * dirs_b).sum(dim=-1).cpu().float().numpy()
    return np.clip(sims, -1.0, 1.0)


def get_projection_gap(
    model: HookedTransformer,
    harmful_toks: Tensor,
    harmless_toks: Tensor,
    directions: Float[Tensor, "L D"],
    pos: int = -1,
    chunk_size: int = 1,
) -> np.ndarray:
    """
    Delta_l^M = E[h_l^M(x) . r_l^M | x in D_harm]
              - E[h_l^M(x) . r_l^M | x in D_safe]

    Uses M's own refusal directions (not M_S's).
    Returns shape (n_layers,).
    """
    n_layers = model.cfg.n_layers

    def chunked_projections(toks: Tensor) -> list:
        layer_projs = [[] for _ in range(n_layers)]
        for i in range(0, toks.shape[0], chunk_size):
            chunk = toks[i : i + chunk_size]
            _, cache = model.run_with_cache(
                chunk, names_filter=lambda n: "resid_pre" in n
            )
            for layer in range(n_layers):
                proj = (
                    cache["resid_pre", layer][:, pos, :].float()
                    * directions[layer].float()
                ).sum(dim=-1)
                layer_projs[layer].append(proj.cpu())
            del cache
            torch.cuda.empty_cache()
        return [torch.cat(p) for p in layer_projs]

    harm_projs = chunked_projections(harmful_toks)
    safe_projs = chunked_projections(harmless_toks)

    gaps = np.array([
        (harm_projs[layer].mean() - safe_projs[layer].mean()).item()
        for layer in range(n_layers)
    ])
    gc.collect()
    torch.cuda.empty_cache()
    return gaps


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING — exactly the two metrics from the paper section
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cosine_similarity(results: dict, family: str):
    """CosSim_l(M, M_S) per layer — one line per distilled model, Base as reference."""
    base_dirs       = results["Base"]["directions"]
    n_layers        = base_dirs.shape[0]
    layers          = np.arange(n_layers)
    distilled_names = [n for n in results if n != "Base"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(1.0, **STYLE["Base"], alpha=0.7, label="Base (reference)")

    all_sims = []
    for name in distilled_names:
        sims = cosine_sim_per_layer(base_dirs, results[name]["directions"])
        ax.plot(layers, sims, **STYLE[name], marker="o", markersize=3, label=name)
        all_sims.append(sims[1:])   # skip layer 0 (degenerate embedding layer)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Cosine Similarity to Base Refusal Direction", fontsize=10)

    if all_sims:
        y_floor = float(np.concatenate(all_sims).min())
        ax.set_ylim(min(y_floor - 0.03, 0.93), 1.015)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = OUTPUT_DIR / f"cosine_similarity_to_base_{family.replace(' ', '_')}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_projection_gap(results: dict, family: str):
    """Delta_l^M (harmful − harmless projection gap) per layer for every model."""
    n_layers = results["Base"]["directions"].shape[0]
    layers   = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, data in results.items():
        ax.plot(layers, data["gap"], **STYLE[name], marker="o", markersize=3, label=name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean Projection Gap (Harmful − Harmless)", fontsize=10)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = OUTPUT_DIR / f"projection_gap_{family.replace(' ', '_')}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def save_summary_csv(results: dict, family: str, last_n: int = 5):
    """
    Write a summary CSV with mean-over-last-N-layers values for every model.

    Columns:
        Model              — model name
        CosSim to Base ↑   — mean CosSim_l(M, M_S) over last N layers
                             ("—" for Base itself, which is the reference)
        Projection Gap ↑   — mean Delta_l^M over last N layers
                             (higher = stronger harmful/harmless separation)
    """
    base_dirs = results["Base"]["directions"]

    rows = []
    for name, data in results.items():
        if name == "Base":
            cossim_str = "—"
        else:
            sims = cosine_sim_per_layer(base_dirs, data["directions"])
            cossim_str = f"{sims[-last_n:].mean():.4f}"

        gap_mean = data["gap"][-last_n:].mean()
        rows.append({
            "Model":              name,
            "CosSim to Base ↑":  cossim_str,
            "Projection Gap ↑":  f"{gap_mean:.4f}",
        })

    out = OUTPUT_DIR / f"summary_{family.replace(' ', '_')}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "CosSim to Base ↑", "Projection Gap ↑"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved CSV: {out}")

    # Also echo the table to console for quick inspection
    col_w = max(len(r["Model"]) for r in rows) + 2
    header = f"{'Model':<{col_w}} {'CosSim to Base ↑':>18}  {'Projection Gap ↑':>18}"
    print(f"\n── [{family}] Summary table (mean over last {last_n} layers) ──")
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r['Model']:<{col_w}} {r['CosSim to Base ↑']:>18}  {r['Projection Gap ↑']:>18}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_hooked_model(
    path: str,
    tl_model_name: str,
    base_model_path: str = None,
    device: str = "cuda",
) -> HookedTransformer:
    print(f"  Loading: {path}")
    is_adapter = (Path(path) / "adapter_config.json").exists()

    if is_adapter:
        print(f"  LoRA adapter — merging into base: {base_model_path}")
        hf_model  = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.float16)
        hf_model  = PeftModel.from_pretrained(hf_model, path)
        hf_model  = hf_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        hf_model  = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(path)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        tl_model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float16,
        default_padding_side="left",
        fp16=True,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# RUN FAMILY
# ═══════════════════════════════════════════════════════════════════════════════

def run_family(
    family: str,
    model_configs: dict,
    tl_model_name: str,
    chat_template: str,
    harmful_train: List[str],
    harmless_train: List[str],
    device: str = "cuda",
):
    print(f"\n{'='*60}\n  Family: {family}  |  Device: {device}\n{'='*60}")

    results = {}
    base_model_path = model_configs["Base"]

    for model_name, model_path in model_configs.items():
        print(f"\n  -- {model_name} --")
        model = load_hooked_model(model_path, tl_model_name,
                                  base_model_path=base_model_path, device=device)

        tokenize_fn = functools.partial(
            tokenize_instructions,
            tokenizer=model.tokenizer,
            template=chat_template,
        )

        harmful_toks  = tokenize_fn(instructions=harmful_train[:N_INST_TRAIN]).to(device)
        harmless_toks = tokenize_fn(instructions=harmless_train[:N_INST_TRAIN]).to(device)

        print("  Computing refusal directions (all layers, pos=-1, resid_pre)...")
        directions = compute_refusal_directions_all_layers(
            model, harmful_toks, harmless_toks, pos=-1
        )

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print("  Computing projection gap Delta_l^M ...")
        gap = get_projection_gap(model, harmful_toks, harmless_toks, directions, pos=-1)

        results[model_name] = {
            "directions": directions.cpu(),
            "gap":        gap,
        }

        print(f"  Layers: {directions.shape[0]} | mean gap (last 5): {gap[-5:].mean():.4f}")

        del model, harmful_toks, harmless_toks, directions
        gc.collect()
        torch.cuda.empty_cache()

    plot_cosine_similarity(results, family)
    plot_projection_gap(results, family)
    save_summary_csv(results, family)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    n_gpus = torch.cuda.device_count()
    llama_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gemma_device = "cuda:1" if n_gpus >= 2 else llama_device
    qwen_device = "cuda:2" if n_gpus >= 3 else llama_device
    print(f"GPUs available: {n_gpus}  |  Llama→{llama_device}  Gemma→{gemma_device}  Qwen→{qwen_device}")

    print("Loading datasets...")
    harmful_train,  _ = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions()
    print(f"  Harmful train: {len(harmful_train)} | Harmless train: {len(harmless_train)}")

    run_family(
        family         = "Llama-3.2-3B",
        model_configs  = LLAMA_CONFIGS,
        tl_model_name  = LLAMA_TL_NAME,
        chat_template  = LLAMA_CHAT_TEMPLATE,
        harmful_train  = harmful_train,
        harmless_train = harmless_train,
        device         = llama_device,
    )

    run_family(
        family         = "Gemma-2-2B",
        model_configs  = GEMMA_CONFIGS,
        tl_model_name  = GEMMA_TL_NAME,
        chat_template  = GEMMA_CHAT_TEMPLATE,
        harmful_train  = harmful_train,
        harmless_train = harmless_train,
        device         = gemma_device,
    )

    run_family(
        family         = "Qwen-2.5-3B",
        model_configs  = QWEN_CONFIGS,
        tl_model_name  = QWEN_TL_NAME,
        chat_template  = QWEN_CHAT_TEMPLATE,
        harmful_train  = harmful_train,
        harmless_train = harmless_train,
        device         = qwen_device,
    )


if __name__ == "__main__":
    main()
