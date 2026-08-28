import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


# CONFIG

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).parent / "outputs" / "pca_layerwise_revised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_MIX_PREFIX = "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/mixing/v2/kd/0.5"
_MIX_SUFFIX = "teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1"

LLAMA_BASE_MODEL = "/media/drive2/models/llama-models/Llama-3.2-3B-Instruct"
GEMMA_BASE_MODEL = "/media/drive2/models/gemma-models/gemma-2-2b-it"
QWEN_BASE_MODEL = "/media/drive2/models/qwen-models/Qwen2.5-3B-Instruct"

LLAMA_CONFIGS = {
    "Base": None,
    "KD-Chat": "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/llama-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/llama-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_204449/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_071424/1424",
    "KD-NonChat": "/media/drive2/distillation/safety-utility-pg/results/llama3/train/kd/chat/8-3B/dolly-prompt/kd/0.5/teacher-base/llama-8B/teacher-adapter/sft/llama8b-it/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20251224_154218/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251225_120349/1246",
    "KD-Mix": f"{_MIX_PREFIX}/{_MIX_SUFFIX}/20260216_115557/1246",
    "KD-Mix-C0.6": f"{_MIX_PREFIX}/curriculum_0.6/{_MIX_SUFFIX}/20260418_102456/1424",
}

GEMMA_CONFIGS = {
    "Base": None,
    "KD-Chat": "/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/gemma-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/gemma-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251222_165206/1424/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_141225/1246",
    "KD-NonChat": "/media/drive2/distillation/safety-utility-pg/results/gemma2/train/kd/chat/9-2B/dolly-prompt/kd/0.5/teacher-base/gemma-2-9b-it/teacher-adapter/sft/gemma-2-9B-it/dolly-prompt/e8-bs1-lr2e-05-G16-N4-NN1-lora-8-16-0.1/20251221_124202/712/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20251223_074250/1424",
}

QWEN_CONFIGS = {
    "Base": None,
    "KD-Chat": "/media/drive2/distillation/safety-utility-pg/results/qwen2/train/kd/chat/7-3B/dolly/qwen-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/qwen-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_065458/1068/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_024037/1424",
    "KD-NonChat": "/media/drive2/distillation/safety-utility-pg/results/qwen2/train/kd/chat/7-3B/dolly/dolly-prompt/0.5/teacher-base/qwen2.5-7B-it/teacher-adapter/sft/qwen2.5-7B-it/dolly/dolly-prompt/e8-bs2-lr2e-05-G8-N4-NN1-lora-8-16-0.1/20260519_184644/1246/e8-bs1-lr2e-05-G16-N4-NN1-kd0.5-lora-8-16-0.1/20260520_085107/1424",
}

FAMILIES = {
    "Llama-3.2-3B": (LLAMA_BASE_MODEL, LLAMA_CONFIGS),
    "Gemma-2-2B": (GEMMA_BASE_MODEL, GEMMA_CONFIGS),
    "Qwen-2.5-3B": (QWEN_BASE_MODEL, QWEN_CONFIGS),
}

N_HARMFUL = 200
N_HARMLESS = 200
MAX_PROMPT_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = {
    "Harmful": "#d64b3c",
    "Harmless": "#2f6fb0",
}


# DATA

def load_harmful(path: Path, n: int):
    questions = []
    with path.open() as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return [q["turns"][0] for q in questions[:n]]


def get_prompts():
    harmful_path = PACKAGE_ROOT / "data" / "sorry_bench" / "question.jsonl"
    harmful = load_harmful(harmful_path, N_HARMFUL)
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    harmless = [row["instruction"] for row in dolly.select(range(N_HARMLESS))]
    return harmful, harmless


# MODEL + ACTIVATION EXTRACTION

def load_model(base_model_path: str, adapter_path=None):
    print(f"  Base: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if adapter_path is not None:
        print(f"  Adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def extract_all_layers(model, tokenizer, prompts: list) -> np.ndarray:
    all_acts = []

    for i, text in enumerate(prompts):
        inputs = tokenizer(
            format_prompt(tokenizer, text),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
            padding=False,
        ).to(DEVICE)

        out = model(**inputs, output_hidden_states=True)
        acts = np.stack([
            out.hidden_states[layer + 1][0, -1, :].float().cpu().numpy()
            for layer in range(len(out.hidden_states) - 1)
        ])
        all_acts.append(acts)

        del out, inputs
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    return np.stack(all_acts)


# PCA + SCORING

def pca_layer(harmful_acts: np.ndarray, harmless_acts: np.ndarray):
    X = np.concatenate([harmful_acts, harmless_acts], axis=0).astype(np.float32)
    n_harmful = len(harmful_acts)

    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X)
    return (
        X_proj[:n_harmful],
        X_proj[n_harmful:],
        pca.explained_variance_ratio_,
    )


def separation_score(harm_proj: np.ndarray, harmless_proj: np.ndarray) -> float:
    mu_harm = harm_proj[:, 0].mean()
    mu_harmless = harmless_proj[:, 0].mean()
    pooled_std = np.sqrt(
        (harm_proj[:, 0].var() + harmless_proj[:, 0].var()) / 2 + 1e-8
    )
    return float(abs(mu_harm - mu_harmless) / pooled_std)


def score_layers(harm_all: np.ndarray, harmless_all: np.ndarray) -> list:
    scores = []
    for layer in range(harm_all.shape[1]):
        harm_proj, harmless_proj, _ = pca_layer(
            harm_all[:, layer, :],
            harmless_all[:, layer, :],
        )
        scores.append(separation_score(harm_proj, harmless_proj))
    return scores


def select_divergent_layer(scores_by_model: dict, target_model: str) -> dict:
    base_scores = np.asarray(scores_by_model["Base"])
    target_scores = np.asarray(scores_by_model[target_model])
    deltas = np.abs(target_scores - base_scores)
    layer = int(np.argmax(deltas))
    return {
        "selected_by": target_model,
        "layer": layer,
        "base_score": float(base_scores[layer]),
        "target_score": float(target_scores[layer]),
        "abs_delta": float(deltas[layer]),
    }


# PLOTTING

def draw_scatter(ax, result: dict, model_name: str, layer: int, base_score: float):
    harm_all = result["activations"][model_name]["harmful"]
    harmless_all = result["activations"][model_name]["harmless"]
    harm_proj, harmless_proj, var_ratio = pca_layer(
        harm_all[:, layer, :],
        harmless_all[:, layer, :],
    )

    ax.scatter(
        harmless_proj[:, 0],
        harmless_proj[:, 1],
        c=COLORS["Harmless"],
        alpha=0.55,
        s=12,
        label="Harmless",
        rasterized=True,
    )
    ax.scatter(
        harm_proj[:, 0],
        harm_proj[:, 1],
        c=COLORS["Harmful"],
        alpha=0.55,
        s=12,
        label="Harmful",
        rasterized=True,
    )

    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.grid(True, alpha=0.18)
    ax.set_title(
        f"{model_name}\nLayer {layer}",
        fontsize=9,
        fontweight="bold" if model_name != "Base" else "normal",
    )
    ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=7)


def build_layer_figure(result: dict, family: str, layer: int, subtitle: str = None):
    model_order = result["model_order"]
    base_score = result["scores"]["Base"][layer]
    fig_width = max(10, 3.4 * len(model_order))
    fig, axes = plt.subplots(1, len(model_order), figsize=(fig_width, 3.8), squeeze=False)

    for ax, model_name in zip(axes[0], model_order):
        draw_scatter(ax, result, model_name, layer, base_score)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9)

    title = f"{family} | Layer {layer}"
    if subtitle is not None:
        title = f"{title} | {subtitle}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def save_family_layer_pdfs(results_by_family: dict):
    for family, result in results_by_family.items():
        family_dir = OUTPUT_DIR / family
        family_dir.mkdir(parents=True, exist_ok=True)

        n_layers = len(result["scores"]["Base"])
        all_layers_path = family_dir / f"all_layers_{family}.pdf"

        with PdfPages(all_layers_path) as pdf:
            for layer in range(n_layers):
                fig = build_layer_figure(result, family, layer)
                layer_path = family_dir / f"layer_{layer:02d}.pdf"
                fig.savefig(layer_path, dpi=150, bbox_inches="tight")
                pdf.savefig(fig, dpi=150, bbox_inches="tight")
                plt.close(fig)

        print(f"Saved layer PDFs: {family_dir}")
        print(f"Saved all-layers PDF: {all_layers_path}")


def save_selected_layer_pdfs(results_by_family: dict, selection_summary: dict):
    selected_dir = OUTPUT_DIR / "selected_layers"
    selected_dir.mkdir(parents=True, exist_ok=True)

    for family, result in results_by_family.items():
        for target_model in ("KD-Chat", "KD-NonChat"):
            if target_model not in selection_summary[family]:
                continue

            selection = selection_summary[family][target_model]
            layer = selection["layer"]
            subtitle = (
                f"selected by {target_model}, "
                f"|delta|={selection['abs_delta']:.2f}"
            )
            fig = build_layer_figure(result, family, layer, subtitle=subtitle)
            out_path = selected_dir / f"{family}_selected_by_{target_model}.pdf"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved selected-layer PDF: {out_path}")


def save_combined_last_layer_pdf(results_by_family: dict):
    families = ["Llama-3.2-3B", "Gemma-2-2B", "Qwen-2.5-3B"]
    model_order = ["Base", "KD-NonChat", "KD-Chat"]
    fig, axes = plt.subplots(
        len(families),
        len(model_order),
        figsize=(10.2, 11.4),
        squeeze=False,
    )

    for row_idx, family in enumerate(families):
        result = results_by_family[family]
        layer = len(result["scores"]["Base"]) - 1
        base_score = result["scores"]["Base"][layer]

        for col_idx, model_name in enumerate(model_order):
            draw_scatter(axes[row_idx][col_idx], result, model_name, layer, base_score)
            axes[row_idx][col_idx].set_title(
                f"{model_name}\n{family} layer {layer}",
                fontsize=10,
                fontweight="bold"
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = OUTPUT_DIR / "combined_last_layer_all_families.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined last-layer PDF: {out_path}")


# MAIN

def run_family(
    family: str,
    base_model_path: str,
    adapter_configs: dict,
    harmful_prompts: list,
    harmless_prompts: list,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"Running family: {family}")
    print(f"{'=' * 70}")

    scores_by_model = {}
    activations_by_model = {}

    for model_name, adapter_path in adapter_configs.items():
        print(f"\n{family} | {model_name}")
        model, tokenizer = load_model(base_model_path, adapter_path)
        n_layers = model.config.num_hidden_layers

        print(f"  Extracting harmful activations: n={len(harmful_prompts)}, layers={n_layers}")
        harm_all = extract_all_layers(model, tokenizer, harmful_prompts)

        print(f"  Extracting harmless activations: n={len(harmless_prompts)}, layers={n_layers}")
        harmless_all = extract_all_layers(model, tokenizer, harmless_prompts)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  Scoring all layers with plain PCA")
        scores = score_layers(harm_all, harmless_all)
        scores_by_model[model_name] = scores
        activations_by_model[model_name] = {
            "harmful": harm_all,
            "harmless": harmless_all,
        }

        top_layers = sorted(enumerate(scores), key=lambda item: -item[1])[:5]
        print("  Top-5 harmful-vs-harmless PC1 separation layers:")
        for rank, (layer, score) in enumerate(top_layers, 1):
            print(f"    #{rank}: layer {layer:2d}, d={score:.3f}")

    return {
        "scores": scores_by_model,
        "activations": activations_by_model,
        "model_order": list(adapter_configs.keys()),
    }


def build_selection_summary(results_by_family: dict) -> dict:
    summary = {}
    for family, result in results_by_family.items():
        summary[family] = {}
        for target_model in ("KD-Chat", "KD-NonChat"):
            if target_model not in result["scores"]:
                continue
            selected = select_divergent_layer(result["scores"], target_model)
            summary[family][target_model] = selected
            print(
                f"{family} | {target_model}: selected layer {selected['layer']} "
                f"(Base d={selected['base_score']:.3f}, "
                f"{target_model} d={selected['target_score']:.3f}, "
                f"|delta|={selected['abs_delta']:.3f})"
            )

        base_scores = np.asarray(result["scores"]["Base"])
        for model_name, scores in result["scores"].items():
            if model_name == "Base":
                continue
            deltas = np.abs(np.asarray(scores) - base_scores)
            layer = int(np.argmax(deltas))
            summary[family][f"{model_name}_vs_base_max"] = {
                "layer": layer,
                "base_score": float(base_scores[layer]),
                "model_score": float(scores[layer]),
                "abs_delta": float(deltas[layer]),
            }

    summary_path = OUTPUT_DIR / "selection_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved selection summary: {summary_path}")
    return summary


def main():
    print("Loading prompts")
    harmful_prompts, harmless_prompts = get_prompts()
    print(f"  Harmful:  {len(harmful_prompts)}")
    print(f"  Harmless: {len(harmless_prompts)}")

    results_by_family = {}
    for family, (base_model_path, adapter_configs) in FAMILIES.items():
        results_by_family[family] = run_family(
            family=family,
            base_model_path=base_model_path,
            adapter_configs=adapter_configs,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
        )

    # selection_summary = build_selection_summary(results_by_family)
    # save_family_layer_pdfs(results_by_family)
    # save_selected_layer_pdfs(results_by_family, selection_summary)
    save_combined_last_layer_pdf(results_by_family)


if __name__ == "__main__":
    main()
