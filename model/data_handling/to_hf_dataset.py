import json
from datasets import Dataset
from transformers import AutoTokenizer
import tqdm


if __name__ == '__main__':
    with open("../combined_data.json") as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    hf_training_data = {"text": []}
    for key in tqdm.tqdm(list(data.keys())):
        sorted_keys = sorted(data[key]["outputs"], key=lambda x:x[0])
        tokens = tokenizer(data[key]["text"])["input_ids"]
        output_text = ""
        start = 0
        if len(sorted_keys) < 2:
            continue
        for i in range(len(sorted_keys)):
            if sorted_keys[i][0] != 0:
                output_text += tokenizer.decode(tokens[start:sorted_keys[i][0]])
                start = sorted_keys[i][0]
            output_text += " [" + sorted_keys[i][1] + " -> " + str(sorted_keys[i][2]) + "]"
        if start < len(tokens)-1:
            output_text += tokenizer.decode(tokens[start:])
        hf_training_data["text"].append(output_text)
    dataset = Dataset.from_dict(hf_training_data)
    dataset.save_to_disk("../sourceformer-dataset")
    # Show the first 10 examples
    print(dataset[:10])
    dataset.push_to_hub("eerichmond33/sourceformer-dataset")

