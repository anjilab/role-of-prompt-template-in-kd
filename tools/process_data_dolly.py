import multiprocessing
import os
import time
from datetime import datetime
import torch
import json
import sys
import numpy as np
from fractions import Fraction
from data_utils.indexed_dataset import make_builder, best_fitting_dtype
from transformers import AutoTokenizer
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def _select_response(self, raw_response):
        if isinstance(raw_response, list):
            return raw_response[0] if raw_response else ""
        return raw_response

    def _use_curriculum_prompt(self, sample_idx, curriculum_mix_ratio):
        ratio = Fraction(str(curriculum_mix_ratio))
        if ratio < 0 or ratio > 1:
            raise ValueError(f"curriculum_mix_ratio must be between 0 and 1, got {curriculum_mix_ratio}")
        return ((sample_idx + 1) * ratio.numerator // ratio.denominator) > (
            sample_idx * ratio.numerator // ratio.denominator
        )

    def _build_chat_prompt(self, line, apply_system_prompt=False):
        instruction = line.get("instruction", "")
        input_text = line.get("input", "").strip()
        system_prompt = (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information."
        )

        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        if apply_system_prompt:
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        else:
            messages = [
                {"role": "user", "content": user_content},
            ]

        return Encoder.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            date_string= datetime.now().strftime("%d %b %Y")
        )

    def _build_default(self, line):
        instruction = line.get("instruction", "")
        input_text = line.get("input", "").strip()

        if not input_text:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )

        else:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )

        return template.format(instruction=instruction, input=input_text)


    def encode(self, item):
        # Support both plain line and (index, line) tuple for curriculum mixing
        if isinstance(item, tuple):
            sample_idx, line = item
        else:
            sample_idx = None
            line = item
        
        line = json.loads(line)
        mt = getattr(self.args, 'model_type', None)
        
        # Curriculum learning mix: decide between curriculum type and default
        curriculum_mix_ratio = getattr(self.args, 'curriculum_mix_ratio', None)
        if curriculum_mix_ratio is not None and sample_idx is not None:
            # Deterministic fractional schedule: exactly floor(n * ratio)
            # curriculum prompts are selected in the first n samples.
            use_curriculum = self._use_curriculum_prompt(sample_idx, curriculum_mix_ratio)
            prompt_type = mt if use_curriculum else "default"
        else:
            prompt_type = mt

        if prompt_type == "llama3-chat":
            prompt = self._build_chat_prompt(line, apply_system_prompt=True)
        elif prompt_type == "gemma2-chat":
            prompt = self._build_chat_prompt(line)
        elif prompt_type == "qwen2-chat":
            prompt = self._build_chat_prompt(line)
        else:
            prompt = self._build_default(line)
        
        raw_response = line.get("output", "")
        response = self._select_response(raw_response)

        input_text = line.get("input", "").strip()

    
        # Tokenize
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt + response, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length] # New abkd
            # return None, None, None, None, len(line)

        line["output"] = response  
        line["input"] = input_text  
        
        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print("PREPROCESSING-DOLLY")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # If in test mode, load the `valid.jsonl` and create a single `valid` split.
    if getattr(args, 'test', False):
        print("Processing in test mode.", args.data_dir)
        valid_path = os.path.join(args.data_dir, "valid.jsonl")
        print("Loading test data from:", valid_path)
        if not os.path.exists(valid_path):
            raise FileNotFoundError(f"Expected test file not found: {valid_path}")
        with open(valid_path) as f:
            valid_data = f.readlines()

        all_data = {
            "test": valid_data
        }
    else:
        with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
            raw_data = f.readlines()

        if args.dev_num > 0:
            all_data = {
                "valid": raw_data[:args.dev_num],
                "train": raw_data[args.dev_num:]
            }
        else:
            all_data = {
                "train": raw_data
            }
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = best_fitting_dtype(len(tokenizer))
    split_id = np.iinfo(dtype).max
    print("dtype:", dtype, "split_id:", split_id)
    for split in all_data:
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # Prepare data: if curriculum mixing, add indices; otherwise just use lines
        curriculum_mix_ratio = getattr(args, 'curriculum_mix_ratio', None)
        if curriculum_mix_ratio is not None:
            print(f"Using curriculum learning mix ratio: {curriculum_mix_ratio}")
            data_to_process = enumerate(all_data[split])
        else:
            data_to_process = all_data[split]

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, data_to_process, chunksize=50)
        # encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=dtype)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [split_id] + response))

            json_file.write(json.dumps({
                "instruction": line["instruction"],
                "prompt": prompt_str,
                "input": line["input"],
                "output": line["output"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()
