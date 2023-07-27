import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import json
import sys
import math
import logging
import signal
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from args import build_parser
from utils.helper import *
from utils.logger import get_logger, print_log, store_results, store_val_results
from dataloader import TextDataset
from model import build_model, train_model, run_validation

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

source_code = '''
def add(num1, num2):
    return num1 + num2

def subtract(num1, num2):
    return num1 - num2

def divide(num1, num2):
    if num2 == 0:
        raise ValueError("Cannot divide by zero.")
    return num1 / num2

def multiply(num1, num2):
    return num1 * num2
    
'''

def main():
    '''Change the debuggers directory'''
    if os.getcwd() != '/home/eli.richmond/SVAMP/code/toolformer':
        os.chdir("/home/eli.richmond/SVAMP/code/toolformer")

    '''read arguments'''
    parser = build_parser()
    config = parser.parse_args()
    logfile = open(os.path.join(log_folder, f'{config.dataset}_1_3.json'), 'w')
    logfile.write(f"Config: {config}\n\n")
    logfile.flush()

    # with open(os.path.join(log_folder, 'the_answer_is.json'), "a") as f:
    #     output_data = json.load(f)

    '''Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    '''GPU initialization'''
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    '''Load the data'''
    test_set = TextDataset(data_path=data_path, dataset=config.dataset,
                            datatype='test', max_length=config.max_length, is_debug=config.debug)
    test_dataloader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
    
    '''Load model & tokenizer'''
    # tokenizer = AutoTokenizer.from_pretrained("eerichmond33/v29")
    # model = AutoModelForCausalLM.from_pretrained("eerichmond33/v29").to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        # revision="float16",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0
    )
    # Set "]" as an end of sentence token
    model.config.eos_token_id = tokenizer("]")["input_ids"][0]

    '''Generate answers'''
    counter = 0
    print(len(test_dataloader))
    print(config.dataset)
    json_data = {}
    questions, gen_eqns, act_eqns, scores = [], [], [], []
    for data in test_dataloader:
        nums = data['nums']
        ans = data['ans']
        eqn = data['eqn']
        if config.grade_disp:
            grade = data['grade']
        if config.type_disp:
            type1 = data['type']
        if config.challenge_disp:
            type1 = data['type']
            var_type = data['var_type']
            annotator = data['annotator']
            alternate = data['alternate']
        # Format question
        ques = data['ques']
        nums_split = nums[0].split()
        for i in range(config.batch_size):
            modified_ques = ques[i]
            for j in range(len(nums_split)):
                modified_ques = modified_ques.replace(f"number{j}", str(nums_split[j]))
            ques[i] = modified_ques + " The answer is"              # <--------------------------------------------------------------------------------------------------
        # Generate
        generated_text = generator(ques, max_new_tokens=30, num_return_sequences=1)
        only_generated_text = generated_text[0][0]["generated_text"].split(modified_ques)[1]
        only_generated_text_no_function_call = ""
        # Check if a function was called and execute it
        if only_generated_text[-1] == "]":
            api_call, api_call_with_brackets = format_api(only_generated_text)
            executed_api = execute_api(api_call)
            if executed_api != "" and executed_api != None:
                executed_api_call = insert_api(executed_api, api_call_with_brackets)
                if executed_api_call != None:
                    only_generated_text = only_generated_text.replace(api_call_with_brackets, executed_api_call)

                    # Continue generation
                    ques[0] = modified_ques + only_generated_text
                    generated_text = generator(ques, max_new_tokens=30, num_return_sequences=1)
                    only_generated_text = generated_text[0][0]["generated_text"].split(modified_ques)[1]
                    only_generated_text_no_function_call = only_generated_text.replace(executed_api_call, "")

        # Write to json
        json_data = {"question": ques[0], "generated_text": only_generated_text, "generated_text_no_function" : only_generated_text_no_function_call, "answer": ans.item(), "equation": eqn[0]}
        # Flush json
        json.dump(json_data, logfile, indent=2)
        logfile.write(",\n")
        logfile.flush()
        json_data = {}
        counter += 1

    logfile.close()

    '''Evaluate answers'''

    return

def insert_api(executed_api, api_call_with_brackets):
    # Loop backwards till ">" is found
    for i in range(len(api_call_with_brackets) - 1, 0, -1):
        if api_call_with_brackets[i] == ">":
            # Grab eveything after ">"
            return api_call_with_brackets[:i + 1] + " " + executed_api + "]"

def execute_api(api_call):
    calculator_api_output = ""
    # Assume the text is worth evaluting if it has add, subtract, multiply, or divide in it.
    if "add" in api_call or "subtract" in api_call or "multiply" in api_call or "divide" in api_call:
        if "input" not in api_call:
            # Call the functions with the generated string
            try:
                # Sometimes it will generate code that hangs out the data generation, so we run it with a timeout time of 20 seconds
                calculator_api_output = evaluate_with_timeout(api_call, 20)

                # CHeck if it is None
                if calculator_api_output is None:
                    raise Exception("Function call returned None")
                # Check if it is a non-number
                elif not isinstance(calculator_api_output, (int, float)):
                    raise Exception("Function call returned non-number")
                
                calculator_api_output = str(calculator_api_output)
            except Exception as e:
                calculator_api_output = ""

    return calculator_api_output

def evaluate_with_timeout(expression, timeout):
    def timeout_handler(signum, frame):
        raise TimeoutError("Evaluation timed out")

    # Set up the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the timeout value

    try:
        result = eval(expression)
        return result
    finally:
        signal.alarm(0)  # Disable the timeout

def format_api(unformatted_api_call):
    # loop backwards till "[" is found
    for i in range(len(unformatted_api_call)-1, -1, -1):
        if unformatted_api_call[i] == "[":
            # Get rid of the front [ and back ]
            api_call = unformatted_api_call[i+1:-1]
            api_call_with_brackets = unformatted_api_call[i:]
            break
    
    # loop backwards till ")" is found
    for i in range(len(api_call)-1, -1, -1):
        if api_call[i] == ")":
            # Grab everything in front of )
            api_call = api_call[:i+1]
            break

    return api_call, api_call_with_brackets

# The source code our model is learning to use
def add(num1, num2):
    return round(num1 + num2, 2)

def subtract(num1, num2):
    return round(num1 - num2, 2)

def divide(num1, num2):
    if num2 == 0:
        raise ValueError("Cannot divide by zero.")
    return round(num1 / num2, 2)

def multiply(num1, num2):
    return round(num1 * num2, 2)

if __name__ == "__main__":
    main()