# Sourceformer

## Abstract

LLMs have been shown to use tools well. By allowing specific tools to increase the capabilities that LLMs struggle with, these models can become much more useful. Previous works use handcrafted examples of simple tool use during self-training. This type of data generation and training is a great compliment to the self-supervised nature of LLMs because most of the generation effort is placed on the LLM. But as the tools become more complex it will become harder to handwrite thorough examples that allow this generation. Today we introduce Sourceformer, which attempts to use a tool in the form of raw source code for self-training and benchmarking during evaluation. We propose a potentially viable method that allows tools to easily grow in complexity and size as the input token sequence to our LLMs inevitably grows. We focus on one tool in particular, a calculator, as a proof of concept for this idea; although, our results are sub par. Across three math benchmarks SVAMP, MAWPS, and ASDiv our model accuracy increases slightly, for some versions, compared to our base model before finetuning.

## Models and Dataset

My models and dataset are avalibale through huggingface.

#### Models
- [sourceformer-epoch1](https://huggingface.co/eerichmond33/sourceformer-epoch1)
- [sourceformer-epoch2](https://huggingface.co/eerichmond33/sourceformer-epoch2)
- [sourceformer-epoch10](https://huggingface.co/eerichmond33/sourceformer-epoch10)
- [sourceformer-epoch30](https://huggingface.co/eerichmond33/sourceformer-epoch30)

#### Dataset
- [sourceformer-dataset](https://huggingface.co/datasets/eerichmond33/sourceformer-dataset)

## Running the code

The codebase for the Sourceformer model is based on two open source repositories.

### /model

[conceptofmind/toolformer](https://github.com/conceptofmind/toolformer) is an open-source implementation of [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) by Meta AI.

I have copied their README.md as it stood when I used it and placed it in [/model.md]

### /benchmarks

[arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) is the [SVAMP](https://arxiv.org/abs/2103.07191), [ASDiv](https://arxiv.org/abs/2106.15772), and [MAWPS](https://aclanthology.org/N16-1136.pdf) math benchmarks. All three are cleaned up and compiled in the SVAMP paper's repository.

I have copied their README.md as it stood when I used it and placed it in [/benchmarks.md]
