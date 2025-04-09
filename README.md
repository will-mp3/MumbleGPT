# DIY-Large-Language-Model

Note: this project is based on the tutorial posted by freeCodeCamp.org (https://www.youtube.com/watch?v=UU1WVnMk4E8). 

Much, but not all, code is original. I do not claim full ownership.

## Setup

This project was developed on an Apple M1 chip running MacOS Sonoma 14.6.1

Development made use of a virtual environment for training and GPU processing.

Because this project was developed in MacOS, Cuda was not utilized.

Instead, mps was used. (more on this later)

To setup the development environment identical to the one in which this LLM was created, follow the following commands:

1. Create a Virtual Environment
- **python -m venv VirtualEnv**
- This creates a virtual environment named VirtualEnv.
- A virtual environment isolates dependencies from the system Python installation, ensuring your project has a controlled environment.
- Be sure to activate the virtual environment using **source VirtualEnv/bin/activate**.

2. Install Required Python Libraries
- **pip install matplotlib numpy pylzma ipykernel jupyter**
- matplotlib → A plotting library for visualization.
- numpy → Provides numerical operations and support for large, multi-dimensional arrays.
- pylzma → A Python library for handling LZMA (compression).
- ipykernel → Allows Jupyter to run different Python environments (kernels).
- jupyter → Installs Jupyter Notebook, a web-based interface for running Python code interactively.
- Note we will not wake use of pylzma due to compatability issues and its existense as a python standard library import, **import lzma**.
  
3. Install PyTorch and Related Libraries
- **pip3 install torch torchvision torchaudio**
- torch → The core PyTorch library for tensor computations and deep learning.
- torchvision → Provides image datasets and pre-processing utilities.
- torchaudio → Provides audio processing tools.
- PyTorch will automatically detect the best backend for execution.
- Since you’re on an M1 Mac, PyTorch should install the mps backend instead of CUDA.
  
4. Create a Jupyter Kernel for the Virtual Environment
- **python -m ipykernel install --user --name=gpu_kernel --display-name "gpu kernel"**
- Creates a Jupyter kernel named gpu_kernel using the virtual environment.
- --display-name "gpu kernel" sets how the kernel appears in Jupyter Notebook.
- After this, you can open Jupyter and select "gpu kernel" as your notebook kernel.

Once completed your machine should be ready for testing and development.

To launch jupyter, use the command **jupyter notebook**.

## Overview

This repo contains multiple models, data files, example work, and data extractors.

All of which are useful and/or were used in the making & understanding of this LLM.

The Bigram Language Model is the foundation for our GPT model and was the first to be created.

The Bigram is trained on the Wizard of Oz .txt file and operates, well, poorly.

It is easy to play with and train the Bigram model and requires no setup further than the above steps.

The GPT model is a little more tedious.

The GPT is trained on the openwebtext corpus which can be downloaded from this link: https://huggingface.co/datasets/Skylion007/openwebtext

You will need the data extractors in this repo to extract the contents from openwebtext's .xz files.

As you may have guessed that is what the GPT model is trained on, just like chatGPT 2.0.

These models are rich with comments, many of which explaining function, purpose, or just general instruction.

Take your time to explore and train models of your own, just remember you need the aforementioned virtual environment and dataset (not included in repo).

Cheers!

Will
