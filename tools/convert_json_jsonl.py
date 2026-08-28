import json
import os

BASE_PATH="/home/minillm_ai_safety"

json_path = os.path.join(BASE_PATH, "data/meta_math/MetaMathQA-40K.json")
jsonl_path = os.path.join(BASE_PATH, "data/meta_math/MetaMathQA-40K.jsonl")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # This loads the entire list of 40,000 examples

# Now write as JSONL (one JSON object per line)
with open(jsonl_path, "w", encoding="utf-8") as f:
    for example in data:
        formatted_example = {
            "query": example["query"].strip(),            # Use question as the prompt
            "output": example["response"].strip()              # Full CoT + final answer
        }
        f.write(json.dumps(formatted_example, ensure_ascii=False) + "\n")

print(f"Converted and saved to {jsonl_path}")