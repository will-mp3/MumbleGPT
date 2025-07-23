# Micro GPT

Note: this project is based on the tutorial posted by freeCodeCamp.org (https://www.youtube.com/watch?v=UU1WVnMk4E8). 

Built for learning purposes. I do not claim full ownership.

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

MumbleGPT is a rendition of a generative pre-trained-transformer with a couple "fun" distinctions.
First, it is NOT a state of the art model, I made this model during finals week senior year on a four year old macbook.
As the name suggests, it does not speak exceptionally well.

Hence the name: MumbleGPT.

While it may pale in comparison to the ever evolving chatbot overlords, it is still quite interesting.
There are four files of note in this repository: training.py, model.py, gui.py, and fix_model.py.
- training.py comes directly from the course linked above and is used to build and train our GPT model, the code is verbose and includes enough comments to learn its inner workings top to bottom.
- model.py is a combination of training.py and my own code, its purpose is to load and deploy our model either through the command line or the included gui.
- gui.py is my own creation and builds upon the model in the form of a gui so that its capability may be better observed (outside of the command line).
- fix_model.py is also my own creation, and was unfortunately necesary due to how the model was originally pickled when created, this script refactors the model so that it may be used in our gui.

This was such a fun project to build and the information I learned from it has been invaluable to me.
Setup will be annoying, and honestly I dont really recommend it.
Instead, check out my demo available on my portoflio site, you will find the link to that on my profiles README.

Will
