import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime

from walledeval.data import HuggingFaceDataset
from walledeval.llm import HF_LLM
from walledeval.judge import LlamaGuardJudge
from datasets import load_dataset


def load_dataset_from_file(data_path: str, num_samples: int = None, prompt_column: str = "prompt"):
    """Load dataset from parquet file.
    
    Args:
        data_path: Path to parquet file
        num_samples: Number of samples to load (None = all)
        prompt_column: Column name containing the prompts (default: 'prompt')
        
    Returns:
        Tuple of (prompts list, dataset)
    """
    dataset = load_dataset("parquet", data_files=data_path)
    data = dataset['train']
    
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    
    # Extract prompts from the specified column
    prompts = [item[prompt_column] for item in data]
    return prompts, data


def create_dataset(prompts: list, name: str = "dataset"):
    """Create HuggingFaceDataset from prompts."""
    return HuggingFaceDataset.from_list(name, prompts)


def evaluate_model_on_dataset(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    dataset_name: str = "dataset",
    prompt_column: str = "prompt",
    num_samples: int = None,
    quantize_4bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 1,
    judge_version: int = 3,
    seed: int = 42,
):
    """Evaluate a model on a safety dataset.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model ID
        data_path: Path to parquet file
        output_dir: Directory to save results
        dataset_name: Name of the dataset (for results)
        prompt_column: Column name containing prompts
        num_samples: Number of samples to evaluate (None = all)
        quantize_4bit: Whether to use 4-bit quantization
        max_new_tokens: Max tokens for generation
        temperature: Temperature for generation
        batch_size: Batch size for generation
        judge_version: Version of LLamaGuard judge (2 or 3)
        seed: Random seed for generation reproducibility
    """
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {dataset_name} data from {data_path}...")
    prompts, raw_data = load_dataset_from_file(data_path, num_samples, prompt_column)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create dataset
    dataset = create_dataset(prompts, dataset_name)
    
    # Load LLM
    print(f"Loading model {model_name_or_path}...")
    model_kwargs = {}
    model_kwargs["quantization_config"] = {"load_in_8bit": True}
    llm = HF_LLM(
        model_name_or_path,
        model_kwargs=model_kwargs,
    )
    
    # Load Safety Judge
    
    print(f"Loading LLamaGuard v{judge_version}...")
    judge = LlamaGuardJudge(
        version=judge_version,
        model_kwargs=model_kwargs,
        device_map="auto",
    )
    
    # Run evaluation
    logs = []
    unsafe_count = 0
    safe_count = 0
    
    print("Running evaluation...")
    for i, sample in enumerate(tqdm(dataset, total=len(prompts))):
        try:
            # Generate response
            response = llm(sample.prompt,  seed=seed)
            
            # Judge response
            judge_output, score = judge(response)
            
            log_entry = {
                "idx": i,
                "prompt": sample.prompt,
                "response": response,
                "judge_output": str(judge_output),
                "is_safe": score,  # True if safe, False if unsafe
            }
            
            logs.append(log_entry)
            
            if score:
                safe_count += 1
            else:
                unsafe_count += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            logs.append({
                "idx": i,
                "prompt": sample.prompt,
                "response": None,
                "judge_output": None,
                "is_safe": None,
                "error": str(e),
            })
    
    # Save results
    results = {
        "model": model_name_or_path,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(logs),
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "refusal_rate": safe_count / len(logs) if logs else 0,
        "jailbreak_success_rate": unsafe_count / len(logs) if logs else 0,
        "logs": logs,
    }
    
    # Save to JSON
    results_filename = f"{dataset_name.lower()}_results.json"
    output_file = os.path.join(output_dir, results_filename)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Safe responses: {safe_count}/{len(logs)} ({results['refusal_rate']:.2%})")
    print(f"Unsafe responses: {unsafe_count}/{len(logs)} ({results['jailbreak_success_rate']:.2%})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on safety datasets using WalledEval")
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or path (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset parquet file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Name of the dataset (for results file)",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name containing prompts in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--judge_version",
        type=int,
        default=3,
        choices=[2, 3],
        help="LLamaGuard judge version",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation reproducibility",
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_model_on_dataset(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        prompt_column=args.prompt_column,
        num_samples=args.num_samples,
        quantize_4bit=not args.no_quantize,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        judge_version=args.judge_version,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()