import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM, AutoModel

device = torch.device(0 if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-7b-1.1")

# model = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-7b-1.1")
# tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-3B")

# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-3B")
# tokenizer = AutoTokenizer.from_pretrained("chainyo/alpaca-lora-7b")

# model = AutoModelForCausalLM.from_pretrained("chainyo/alpaca-lora-7b")

# tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-3b")

# model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-3b")
tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-7B-HF")

model = AutoModelForCausalLM.from_pretrained("TheBloke/koala-7B-HF")
model.to(device)



generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
generator.device = device

# Generate text using the generator
prompt = "What is up my brotha?"
generated_text = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print("Generated Text:")
for sequence in generated_text:
    print(sequence["generated_text"])
