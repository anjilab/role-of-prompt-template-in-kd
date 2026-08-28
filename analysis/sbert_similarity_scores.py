
import json
import statistics
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    torch = None
    SentenceTransformer = None
    MISSING_DEPENDENCY = exc
else:
    MISSING_DEPENDENCY = None


CKPT_NAME = "llama"

SEEDS = [10, 20, 30, 40, 50]

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(
    "/media/drive2/distillation/safety-utility-pg/results/llama3/eval_main"
)
DATASET_NAME = "dolly-1024"
TRUE_DATA_PATH = BASE_DIR / "processed_data" / "dolly" / "full" / "llama3" / "test.jsonl"
SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_generated_answers(file_path):
    """Load generated answers from an answers.jsonl file."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    except Exception:
        return None


def load_true_answers(file_path):
    """Load ground-truth references from a validation JSONL file."""
    with open(file_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    answers = []
    for row in rows:
        output = row["output"]
        if isinstance(output, list):
            output = output[0]
        answers.append(output)

    return answers


def find_latest_answers(config_dir, seed):
    """Find the newest answers.jsonl for one config and seed."""
    seed_dir = config_dir / str(seed)
    if not seed_dir.exists():
        return None

    answer_files = sorted(seed_dir.glob("*/answers.jsonl"))
    if not answer_files:
        return None

    return answer_files[-1]


def discover_config_dirs(results_dir, dataset_name):
    """Discover model config directories directly under matching dataset dirs."""
    config_dirs = []

    for dataset_dir in results_dir.glob(f"**/{dataset_name}"):
        if not dataset_dir.is_dir():
            continue

        for child in sorted(dataset_dir.iterdir()):
            if child.is_dir():
                config_dirs.append(child)

    return sorted(set(config_dirs), key=lambda path: str(path))


def encode_texts(model, texts):
    """Encode texts with normalized SBERT embeddings."""
    with torch.no_grad():
        return model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )


def mean_cosine_similarity(reference_embeddings, generated_embeddings):
    """Compute mean row-wise cosine similarity for normalized embeddings."""
    return torch.sum(reference_embeddings * generated_embeddings, dim=1).mean().item()


def compute_diversity(embeddings_by_seed):
    """
    Compute inter-seed diversity as 1 - average pairwise similarity.
    Higher means outputs vary more across seeds.
    """
    pair_sims = []

    for i in range(len(embeddings_by_seed)):
        for j in range(i + 1, len(embeddings_by_seed)):
            sim = mean_cosine_similarity(embeddings_by_seed[i], embeddings_by_seed[j])
            pair_sims.append(1 - sim)

    if not pair_sims:
        return 0.0

    return statistics.mean(pair_sims) * 100


def main():
    if MISSING_DEPENDENCY is not None:
        raise SystemExit(
            "Missing required dependency for SBERT scoring: "
            f"{MISSING_DEPENDENCY.name}. Run this script in the environment "
            "that has torch and sentence-transformers installed."
        )

    print("\nLoading SentenceTransformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    print(
        f"\n{'Configuration':<45} | "
        f"{'True Sim by Seed':<48} | "
        f"{'Mean':<8} | "
        f"{'Div.'}"
    )
    print("-" * 125)

    if not TRUE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference file not found: {TRUE_DATA_PATH}")

    true_answers = load_true_answers(TRUE_DATA_PATH)
    num_references = len(true_answers)
    true_embeddings = encode_texts(model, true_answers)

    config_dirs = discover_config_dirs(RESULTS_DIR, DATASET_NAME)
    if not config_dirs:
        print(f"No configs found under {RESULTS_DIR} for dataset {DATASET_NAME}")
        return

    for config_dir in config_dirs:
        answers_by_seed = []

        for seed in SEEDS:
            file_path = find_latest_answers(config_dir, seed)
            if file_path is None:
                answers_by_seed = []
                break

            answers = load_generated_answers(file_path)
            if answers is None or len(answers) != num_references:
                answers_by_seed = []
                break

            answers_by_seed.append(answers)

        if not answers_by_seed:
            continue

        embeddings_by_seed = [encode_texts(model, answers) for answers in answers_by_seed]

        true_similarities = [
            mean_cosine_similarity(true_embeddings, embeddings) * 100
            for embeddings in embeddings_by_seed
        ]

        mean_true_sim = statistics.mean(true_similarities)
        diversity_score = compute_diversity(embeddings_by_seed)

        label = str(config_dir.relative_to(RESULTS_DIR))
        tuple_str = "(" + ", ".join(f"{score:.4f}" for score in true_similarities) + ")"

        print(
            f"{label:<45} | "
            f"{tuple_str:<48} | "
            f"{mean_true_sim:.2f} | "
            f"{diversity_score:.2f}"
        )


if __name__ == "__main__":
    main()
