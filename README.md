# DIY-Large-Language-Model

Note: this project is based on the tutorial posted by freeCodeCamp.org (https://www.youtube.com/watch?v=UU1WVnMk4E8). 

While much of the setup and code is original I do not claim full ownership, this project is for learning purposes only.

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
- After this, you can open Jupyter and select "gpu kernel" in running notebooks.

Once completed your machine should be ready for testing and development.

To launch jupyter, use the command **jupyter notebook**.
