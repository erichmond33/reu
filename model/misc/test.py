import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("eerichmond33/finetune_toolformer_v0")
model = AutoModelForCausalLM.from_pretrained("eerichmond33/finetune_toolformer_v0")



generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)


# Generate text using the generator
prompt = "Paco had 26 salty cookies and 17 sweet cookies. He ate 14 sweet cookies and 9 salty cookies. How many salty cookies did Paco have left? The answer is [   "
generated_text = generator(prompt, num_return_sequences=1, max_length=200, top_k=50, top_p=0.95, temperature=0.8)

# Print the generated text
print("Generated Text:")
for sequence in generated_text:
    print(sequence["generated_text"])



# device = torch.device(0 if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch3")
# model = AutoModelForCausalLM.from_pretrained(
#     r"dmayhem93/toolformer_v0_epoch2",
#     torch_dtype=torch.float32,
#     # low_cpu_mem_usage=True,
# ).cuda()
# generator = pipeline(
#     "text-generation", model=model, tokenizer=tokenizer, device=0
# )
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

calculator_prompt = """
Your task is to add calculator function calls to a piece of text. The calls should help you get information required to complete the text. You can call a function by writing "[functionName(arg1, arg2)]". Here are the available calculator functions:

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

Here are some examples of calculator function calls:
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [add(18, multiply(12, 3))] 54.
Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
Output: The population is 658,893 people. This is 11.4% of the national average of [divide(658,893, 11.4%)] 5,763,868 people.
Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [divide(723, 252)] 2.87 per match). This is twenty goals more than the [subtract(723, 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [subtract(2011, 1994)] 17 years.
Input: From this, we have 4 * 30 minutes = 120 minutes.
Output: From this, we have 4 * 30 minutes = [multiply(4, 30)] 120 minutes.
Input: x
Output:
"""

# How many tokens is my prompt?
prompt_tokens = tokenizer.encode(calculator_prompt)
print("Prompt length:", len(prompt_tokens))








# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model.to(device)