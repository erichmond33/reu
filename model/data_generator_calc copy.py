import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import transformers
from datasets import load_dataset
from prompts import retrieval_prompt
from data_generation.retrieval import RetrievalPostprocessing
from data_generation.calendar import CalendarPostprocessing
from data_generation.calculator import CalculatorPostprocessing, apply_heuristics, apply_heuristics
from data_generation.api_checker import check_apis_available
import json
import time
import argparse
import re
import accelerate

if __name__ == "__main__":
    device = torch.device("cuda:1")

    # Just getting the args
    parser = argparse.ArgumentParser(description='do some continuations')
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument("--num_devices", type=int, default=1)
    args = parser.parse_args()

    # Tokenizer
    gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # Some kind of retrieval prompt?
    prompt_tokens = gpt_tokenizer(retrieval_prompt, return_tensors="pt")["input_ids"]
    # The start & end tokens for our API calls
    start_tokens = [
        gpt_tokenizer("[")["input_ids"][0],
        gpt_tokenizer(" [")["input_ids"][0],
    ]
    end_tokens = [
        gpt_tokenizer("]")["input_ids"][0],
        gpt_tokenizer(" ]")["input_ids"][0],
    ]  # TODO: keep second?
    # calculator.py
    calculator_api_handler = CalculatorPostprocessing(start_tokens, end_tokens)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        # revision="float16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    # Configure generation to end when <api_end> is generated (end_tokens[1] almost never gets called)
    model.config.eos_token_id = end_tokens[0]
    # Data
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    filtered_dataset = dataset.filter(lambda example: apply_heuristics(example, gpt_tokenizer, print_heuristics=False))
    iter_data = iter(filtered_dataset)
    # Vars
    calculator_api_avalability = True
    counter = 0
    file_counter = 0
    found_examples = 0
    output_dataset = list()
    start_time = time.process_time()
    num_examples = int(90000.0/float(args.num_devices))
    start_count = -1
    # All for num_examples & start_count to pick up where you left off
    if os.path.isfile(f"calc_data_{args.device_id}.json"):
        with open(f"calc_data_{args.device_id}.json") as f:
            output_dataset = json.load(f)
            start_count = output_dataset[-1]['file_index']
            for item in output_dataset:
                num_examples -= len(item['calculator_outputs'])
    i = 0
    # Loop until enough data is found
    while found_examples < num_examples:
        data = next(iter_data)
        # Increment file_counter to pick up where you left off
        if file_counter < start_count:
            file_counter += 1
            continue
        if i < 2090:
            if i % 100 == 0:
                print(i)
            i += 1
            continue

        # if file_counter % args.num_devices != args.device_id:
        #     file_counter += 1
        #     continue
        # Visulize the data example
        apply_heuristics(data, gpt_tokenizer, print_heuristics=False)
        # Make sure calc api works
        # available = check_apis_available(data, gpt_tokenizer)
        # calculator_api_avalability = available.calculator
        if calculator_api_avalability:
            # Parse the data
            data_outputs = calculator_api_handler.parse_article(data, model, gpt_tokenizer, device)
            # If no data is found, print found: 0
            if len(data_outputs) == 0:
                eta_s = (num_examples - found_examples) * (time.process_time() - start_time) / max(1, found_examples)
                eta_m = eta_s // 60
                eta_h = eta_m // 60
                eta_m = eta_m - (eta_h * 60)
                eta_s = eta_s - ((eta_m * 60) + (eta_h * 60 * 60))
                print(f"device {args.device_id} Found: {found_examples}/{num_examples}, ETA: {eta_h}H:{eta_m}M:{eta_s}s")
                continue
            # else: add data, write it, update counters
            output_dataset.append(
                {
                    "file_index": file_counter,
                    "text": data["text"],
                    "calculator_outputs": data_outputs
                }
            )
            prev_found = found_examples
            found_examples += len(output_dataset[-1]["calculator_outputs"])
            eta_s = (num_examples - found_examples) * (time.process_time()-start_time) / max(1, found_examples)
            eta_m = eta_s // 60
            eta_h = eta_m // 60
            eta_m = eta_m - (eta_h*60)
            eta_s = eta_s - ((eta_m*60) + (eta_h*60*60))
            print(f"device {args.device_id} Found: {found_examples}/{num_examples}, ETA: {eta_h}H:{eta_m}M:{eta_s}s")
            if found_examples//1 > prev_found//1:
                with open(f"calc_data_{args.device_id}.json", 'w') as f:
                    json.dump(output_dataset, f, indent=2)
            counter += 1
        file_counter += 1
    with open(f"calc_data_{args.device_id}.json", 'w') as f:
        json.dump(output_dataset, f, indent=2)
