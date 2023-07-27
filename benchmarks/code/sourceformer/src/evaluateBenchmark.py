import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import json
import re
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

def main():
    '''Change the debuggers directory'''
    if os.getcwd() != '/home/eli.richmond/SVAMP/code/toolformer':
        os.chdir("/home/eli.richmond/SVAMP/code/toolformer")

    '''read arguments'''
    parser = build_parser()
    config = parser.parse_args()

    '''Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    '''Load the data'''
    with open(os.path.join(log_folder, f'{config.dataset}.json'), "r") as f:
        output_data = json.load(f)

    '''Evaluate answers'''
    if config.just_write_stats == False:
        for data in output_data:
            if data["generated_text_no_function"] != "":
                text = data["generated_text_no_function"]
            else:
                text = data["generated_text"]
                
            # Find the first number in text
            numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
            numbers = [number.replace(",", "") for number in numbers] 
            # See if = is in the text
            if "=" in text:
                data["correct"] = "flagged"
            elif len(numbers) > 0:
                data["our_answer"] = numbers[0]
                if float(numbers[0]) == data["answer"]:
                    data["correct"] = "true"
                else:
                    data["correct"] = "false"
            else:
                data["correct"] = "false"

    '''Evaluate the stats'''
    true = 0
    false = 0
    flagged = 0
    total = 0
    for data in output_data:
        total += 1
        if data["correct"] == "true":
            true += 1
        elif data["correct"] == "false":
            false += 1
        elif data["correct"] == "flagged":
            flagged += 1

    percent_correct = true / total
    percent_incorrect = false / total

    '''Write it'''
    with open(os.path.join(log_folder, f'{config.dataset}.json'), "w") as f:
        json.dump(output_data, f, indent=2)

    with open(os.path.join(log_folder, f'{config.dataset}.txt'), "a") as f:
        if config.just_write_stats == False:
            f.write(f"** Uncorrected Results **\n")
        else:
            f.write(f"** Corrected Results **\n")

        f.write(f"Total: {total}\n")
        f.write(f"Correct: {true}\n")
        f.write(f"Incorrect: {false}\n")
        f.write(f"Flagged: {flagged}\n")
        f.write(f"Percent Correct: {percent_correct}\n")
        f.write(f"Percent Incorrect: {percent_incorrect}\n")

    return

if __name__ == "__main__":
    main()