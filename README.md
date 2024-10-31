## Introduction

This is a forked implementation of the original [SELFRec](https://github.com/Coder-Yu/SELFRec) by Coder-Yu.

The main changes are:
- Support multi-modal datasets
- Use Large Language Models to enhance presentations
- Update some deprecated APIs
- Improve code readability by type hint and docstrings
- Refactor some modules

## Environment

The Python and package managers version are:
- Python 3.9.19
- mamba 1.5.8
- conda 24.3.0

Use conda/mamba to install the environment (Recommend):
```bash
mamba create -n selfrec python=3.9
mamba activate selfrec
```

Or use pip:

```bash
pip install -r requirements.txt
```

The `env.full.yml` is exported by `mamba env export`. Do not use this to build the environment, for reference only!

## Pre-trained models

- Vision: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- Text: [bennegeek/stella_en_1.5B_v5](https://huggingface.co/bennegeek/stella_en_1.5B_v5)

> [!warning]
> Use [bennegeek/stella_en_1.5B_v5](https://huggingface.co/bennegeek/stella_en_1.5B_v5) to generate text embeddings requires `flash-attn` which needs a CUDA version of at least 11.6, as indicated by `nvcc -V`.