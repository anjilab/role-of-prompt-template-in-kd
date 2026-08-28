import pyarrow.parquet as pq
import json
import os

BASE_PATH="/home/minillm_ai_safety"

# Input: GSM8K-Platinum parquet file
# parquet_path = os.path.join(BASE_PATH, "data/gsm8k-platinum-test/main/test-00000-of-00001.parquet")
# Input: PubMedQA parquet file
parquet_path = os.path.join(BASE_PATH, "data/pubmedqa/pqa_artificial/train-00000-of-00001.parquet")

# Output: Consistent JSONL with query/response format
jsonl_path = os.path.join(BASE_PATH, "data/meta_math/test.jsonl")  # Renamed for clarity

print("Converting:")
print(f"  From: {parquet_path}")
print(f"  To:   {jsonl_path}")

# Read Parquet file
table = pq.read_table(parquet_path)
data = table.to_pylist()  # List of dicts: [{'question': ..., 'answer': ...}, ...]

# Convert and write as JSONL with consistent keys
with open(jsonl_path, "w", encoding="utf-8") as f:
    for example in data:
        formatted_example = {
            "query": example["question"].strip(),           # Use question as the prompt
            "output": example["answer"].strip()              # Full CoT + final answer
        }
        f.write(json.dumps(formatted_example, ensure_ascii=False) + "\n")

print(f"Successfully converted {len(data)} examples to {jsonl_path}")