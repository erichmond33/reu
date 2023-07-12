import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from tools import Calculator
from prompts import calculator_prompt
from typing import List
from data_generation.base_api import APICallPostprocessing
import dateutil.parser as dparser
import random
import re
import signal


# TODO: Per API?
MAX_BATCH_SIZE = 1  # My 3090 is weak 😔
N = 128  # SEQ Len
M = 16  # Min Loss Span To Consider
MAX_LEN = 1024  # Maximum calculator length

# The source code our model is learning to use
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

def infinite_loop():
    while True:
        pass


class CalculatorPostprocessing(APICallPostprocessing):
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.0,
        # api_text: str = "Calculator(",
        api_text: str = "",
        k_values: int = 20,
        m_generations: int = 10
    ):
        self.calculator = Calculator
        super().__init__(start_tokens, end_tokens, minimum_percentage, api_text, k_values, m_generations)

    def add_api_calls(
        self,
        candidate: int,
        outputs: dict,
        texts_to_test: List[str],
        tokenizer: PreTrainedTokenizerBase,
        input_tokens: torch.Tensor,
        input_start: int,
        nums_to_keep: List[int],
        base_loss: float,
        device,
        *args,
        **kwargs
    ):
        generated_texts = list()
        max_token_len = N
        max_token_len_base = N
        for j in range(len(outputs)):
            # Remove everything before the <API> call to Calculator
            outputs[j]["Calculator"] = outputs[j]["generated_text"].replace(
                texts_to_test[candidate], ""
            )
            # Remove everything before the entire generated sentence
            outputs[j]["Generated"] = outputs[j]["generated_text"].split("Output:")[-1]
            # Check if </API> exsists
            if "]" in outputs[j]["Calculator"]:
                # Remove everything but the math inside the <API> call & potentially the trailing ')'
                outputs[j]["Calculator"] = (
                    outputs[j]["Calculator"].replace("Calculator(", "").split("]")[0]
                )
                outputs[j]["Calculator_text"] = outputs[j]["Calculator"].lower()
                # Assume the text is worth evaluting if it has add, subtract, multiply, or divide in it.
                if "add" in outputs[j]["Calculator_text"] or "subtract" in outputs[j]["Calculator_text"] or "multiply" in outputs[j]["Calculator_text"] or "divide" in outputs[j]["Calculator_text"]:
                    if "input" in outputs[j]["Calculator_text"]:
                        continue

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

                    # Call the functions with the generated string
                    try:
                        # Sometimes it will generate code that hangs out the data generation, so we run it with a timeout time of 20 seconds
                        calculator_api_output = evaluate_with_timeout(outputs[j]["Calculator_text"], 20)
                        outputs[j]["Calculator"] = str(calculator_api_output)

                        # CHeck if it is None
                        if calculator_api_output is None:
                            raise Exception("Function call returned None")
                        # Check if it is a non-number
                        elif not isinstance(calculator_api_output, (int, float)):
                            raise Exception("Function call returned non-number")
                    except Exception as e:
                        continue

                    # Make a list including the entire <API> and the executed api answer
                    outputs[j]["Calculator_output"] = [outputs[j]["Calculator_text"], str(calculator_api_output)]
                    # Tokenize the entire <API> call
                    base_inputs = tokenizer(
                        "[" + outputs[j]["Calculator_text"] + "]" + " ",
                        return_tensors="pt",
                    )["input_ids"].to(device)
                    # Format the entire <API> and executed api answer
                    outputs[j]["Calculator_text"] = (
                        "["
                        + outputs[j]["Calculator_text"]
                        + "->"
                        + str(calculator_api_output)
                        + "]"
                    )
                    # Tokenize the formatted entire <API> call and the executed api answer
                    test_inputs = tokenizer(
                        outputs[j]["Calculator_text"] + " ",
                        return_tensors="pt",
                    )["input_ids"].to(device)
                    # Add the tokenized enitre api call and answer to the front of the tokenized generated sequence
                    test_inputs = torch.concat(
                        [
                            test_inputs.to(device),
                            input_tokens[:, input_start:].to(device),
                        ],
                        dim=1,
                    )
                    if test_inputs.shape[1] > MAX_LEN:
                        continue
                    # Add the tokenized enitre api call without the answer to the front of the tokenized generated sequence
                    base_inputs = torch.concat(
                        [
                            base_inputs.to(device),
                            input_tokens[:, input_start:].to(device),
                        ],
                        dim=1,
                    )
                    max_token_len = max(max_token_len, test_inputs.shape[1])
                    max_token_len_base = max(max_token_len_base, test_inputs.shape[1])
                    generated_texts.append(
                        [
                            test_inputs,
                            base_inputs,
                            nums_to_keep[candidate],
                            base_loss,
                            outputs[j],
                            input_tokens.to(device)
                        ]
                    )
        return generated_texts, max_token_len, max_token_len_base
    
    def parse_article(
        # PreTrainedModel & PreTrainedTokenizerBase are just templates
        self, data: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device
    ):
        # Get the tokens of the input data
        outputs = list()
        tokens = tokenizer(data["text"], return_tensors="pt")["input_ids"]
        # This goes through the whole sentence
        for i in range((tokens.shape[1]-1)//N):
            if (N * (i + 1)) > tokens.shape[1]:
                continue
            # This takes the last N (128) tokens
            input_tokens = tokens[:, (-N * (i + 1) - 1) : (-N * (i) - 1)]
            # This is input_tokens shifted right 1 token
            labels = tokens[
                :,
                int(tokens.shape[1] + (-N * (i + 1))) : int(tokens.shape[1] + (-N * i)),
            ]
            # Decode the tokens
            string = tokenizer.decode(input_tokens[0])
            whole_string = tokenizer.decode(tokens[0,:] )
            # This adds the calculator prompt to the input
            model_input = tokenizer(
                calculator_prompt.replace("<REPLACEGPT>", string) + string,
                return_tensors="pt",
            )["input_ids"]
            # This generates five examples with this prompt.
            with torch.no_grad():
                output = model(model_input.to(device)).logits.cpu()[:, -N:]
            new_outputs = self.generate_continuations(
                model_input,
                output,
                labels,
                model,
                tokenizer,
                device
            )
            # This checks if anything was generated & has a score greater than filter threshold
            for output in new_outputs:
                if output is None:
                    continue
                output["index"] += int(tokens.shape[1] + (-N * (i + 1)))
                # filter by score
                # with open("score.txt", "a") as f:
                #     f.write(str(output["Score"]) + "\n\n")

                if output["Score"] > 0.0:
                    outputs.append([output["Score"], output["index"]] + output["Calculator_output"])
            print(f"{i} out of {(tokens.shape[1]-1)//N}")
        return outputs



def apply_heuristics(example, tokenizer, print_heuristics=False):
    text = example["text"]

    # Heuristic (i)
    tokens = tokenizer.tokenize(text)
    window_size = 100
    # Grab a 100 token window and increment by 1 token
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        detokenized_window = tokenizer.convert_tokens_to_string(window)
        # Capturing all numbers within the window i.e. 4 & 4.2 & 1,000
        numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", detokenized_window)
        numbers = [number.replace(",", "") for number in numbers]
        # Check every combination where one of these numbers is the result of applying a mathematical operation to the other two
        if len(numbers) >= 3:
            for j in range(len(numbers) - 2):
                for k in range(j + 1, len(numbers) - 1):
                    for l in range(k + 1, len(numbers)):
                        num1 = float(numbers[j])
                        num2 = float(numbers[k])
                        num3 = float(numbers[l])
                        if (num2 != 0 and num1 / num2 == num3) or (num1 + num2 == num3) or (num1 - num2 == num3) or (num1 * num2 == num3):
                            if print_heuristics:
                                print("Heuristic i:\nData:" + detokenized_window + "\n")
                                print("Equation: " + str(num1) + " ? " + str(num2) + " = " + str(num3))
                                print(f"Numbers: {numbers}\n\n\n")
                            return True
                        
    # Heuristic (ii)
    if re.search(r"(=|equals|equal to|total of|average of)\s*\d+", text):
        if print_heuristics:
            print("Heuristic ii:\n" + text + "\n")
        return True

    # Heuristic (iii)
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    numbers = [number.replace(",", "") for number in numbers]
    if len(numbers) >= 3:
        if random.random() < 0.01:
            if print_heuristics:
                print("Heuristic iii:\n" + text + "\n")
            return True
                        
    return False