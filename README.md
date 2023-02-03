# Overview

This repo is a branch of [nn_pruning](https://github.com/huggingface/nn_pruning) that implements Gradual Random and Uniqueness pruning.

We are still in the process of cleaning up this codebase, however, usage is largely the same as nn_pruning. If you'd like to use this repo, please follow the setup for that one. 

The main differences are:

- Many pieces of the original research report are removed, to avoid confusion with our work
- We add the files [main.py](main.py), [utils.py](utils.py), [data.py](data.py), [beam_decode.py](beam_decode.py) and [model.py](model.py)
- In [model.py](model.py), we copy the model code (GPT2 and GPT-Neo), so that we can store cosine similarity between neurons in the forward pass. This unfortunately results in a large amount of bloated code, as most of it is copy-pasted from HuggingFace.


Please note the files [main.py](main.py), [utils.py](utils.py), and [data.py](data.py) have all become quite bloated due to the addition of more code. They are in need of a slight rework.