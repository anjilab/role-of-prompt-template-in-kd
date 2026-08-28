import argparse
import json
import os
from statistics import mean

from transformers import AutoTokenizer


def percentile(values, pct):
    if not values:
        return 0
    values = sorted(values)
    idx = round((len(values) - 1) * pct / 100)
    return values[idx]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_num"] = line_num
            rows.append(row)
    return rows


def detect_chat_family(prompt):
    if "<|start_header_id|>" in prompt or "<|eot_id|>" in prompt:
        return "llama"
    if "<start_of_turn>" in prompt or "<end_of_turn>" in prompt:
        return "gemma"
    return "unknown"


def has_valid_chat_tail(prompt, family):
    if family == "llama":
        return (
            "<|start_header_id|>assistant<|end_header_id|>" in prompt
            and prompt.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")
        )
    if family == "gemma":
        return prompt.rstrip().endswith("<start_of_turn>model")
    return True


def has_required_chat_markers(prompt, family):
    if family == "llama":
        return all(
            marker in prompt
            for marker in (
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
            )
        )
    if family == "gemma":
        return "<start_of_turn>user" in prompt and "<start_of_turn>model" in prompt
    return True


def output_format_flags(row):
    output = row.get("output", "")
    if isinstance(output, list):
        output = output[0] if output else ""
    output = str(output)
    return {
        "has_hash_answer": "####" in output,
        "has_answer_is": "answer is" in output.lower(),
        "empty_output": not output.strip(),
    }


def summarize_split(rows, tokenizer, max_prompt_length, max_examples):
    prompt_lens = []
    full_lens = []
    response_lens = []
    over_limit = []
    malformed = []
    bad_tail_after_truncation = []
    output_flags = {
        "has_hash_answer": 0,
        "has_answer_is": 0,
        "empty_output": 0,
    }
    family_counts = {}

    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", ""))
        output = row.get("output", "")
        if isinstance(output, list):
            output = output[0] if output else ""
        output = str(output)

        family = detect_chat_family(prompt)
        family_counts[family] = family_counts.get(family, 0) + 1

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = tokenizer.encode(prompt + output, add_special_tokens=False) + [tokenizer.eos_token_id]
        prompt_len = len(prompt_tokens)
        full_len = len(full_tokens)

        prompt_lens.append(prompt_len)
        full_lens.append(full_len)
        response_lens.append(full_len - prompt_len)

        if prompt_len > max_prompt_length:
            truncated_prompt = tokenizer.decode(prompt_tokens[:max_prompt_length])
            over_limit.append((idx, row, prompt_len, full_len, family, truncated_prompt))
            if not has_valid_chat_tail(truncated_prompt, family):
                bad_tail_after_truncation.append((idx, row, prompt_len, family, truncated_prompt))

        if not has_required_chat_markers(prompt, family) or not has_valid_chat_tail(prompt, family):
            malformed.append((idx, row, prompt_len, family, prompt))

        flags = output_format_flags(row)
        for key, value in flags.items():
            output_flags[key] += int(value)

    total = len(rows)
    print(f"rows: {total}")
    print(f"chat_family_counts: {family_counts}")
    print(
        "prompt_tokens: "
        f"mean={mean(prompt_lens):.2f} "
        f"max={max(prompt_lens) if prompt_lens else 0} "
        f"p50={percentile(prompt_lens, 50)} "
        f"p95={percentile(prompt_lens, 95)} "
        f"p99={percentile(prompt_lens, 99)}"
    )
    print(
        "response_tokens: "
        f"mean={mean(response_lens):.2f} "
        f"max={max(response_lens) if response_lens else 0} "
        f"p50={percentile(response_lens, 50)} "
        f"p95={percentile(response_lens, 95)} "
        f"p99={percentile(response_lens, 99)}"
    )
    print(
        "full_tokens: "
        f"mean={mean(full_lens):.2f} "
        f"max={max(full_lens) if full_lens else 0} "
        f"p50={percentile(full_lens, 50)} "
        f"p95={percentile(full_lens, 95)} "
        f"p99={percentile(full_lens, 99)}"
    )
    print(
        f"over_max_prompt_length_{max_prompt_length}: "
        f"{len(over_limit)} / {total} ({100 * len(over_limit) / total if total else 0:.2f}%)"
    )
    print(
        "truncated_prompt_loses_valid_chat_tail: "
        f"{len(bad_tail_after_truncation)} / {total} "
        f"({100 * len(bad_tail_after_truncation) / total if total else 0:.2f}%)"
    )
    print(
        "malformed_original_chat_prompt: "
        f"{len(malformed)} / {total} ({100 * len(malformed) / total if total else 0:.2f}%)"
    )
    for key, count in output_flags.items():
        print(f"output_{key}: {count} / {total} ({100 * count / total if total else 0:.2f}%)")

    print_examples("over-limit examples", over_limit, tokenizer, max_examples)
    print_examples("bad-tail-after-truncation examples", bad_tail_after_truncation, tokenizer, max_examples)
    print_examples("malformed original examples", malformed, tokenizer, max_examples)


def print_examples(title, examples, tokenizer, max_examples):
    if not examples or max_examples <= 0:
        return

    print(f"\n### {title}")
    for example_num, item in enumerate(examples[:max_examples], start=1):
        idx, row, prompt_len, family, text = item[0], item[1], item[2], item[4] if len(item) == 6 else item[3], item[-1]
        output = row.get("output", "")
        if isinstance(output, list):
            output = output[0] if output else ""
        prompt = str(row.get("prompt", ""))
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = tokenizer.encode(prompt + str(output), add_special_tokens=False) + [tokenizer.eos_token_id]

        print(f"\n--- example {example_num} row_index={idx} line={row.get('_line_num')} family={family}")
        print(f"prompt_len={prompt_len} full_len={len(full_tokens)} response_len={len(full_tokens) - len(prompt_tokens)}")
        if "query" in row:
            print(f"query: {str(row['query'])[:500]}")
        elif "instruction" in row:
            print(f"instruction: {str(row['instruction'])[:500]}")
        print("decoded_text:")
        print(text[:3000])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-examples", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for split in args.splits:
        path = os.path.join(args.processed_data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"\n## {split}: missing {path}")
            continue

        print(f"\n## {split}: {path}")
        rows = load_jsonl(path)
        summarize_split(rows, tokenizer, args.max_prompt_length, args.max_examples)


if __name__ == "__main__":
    main()
