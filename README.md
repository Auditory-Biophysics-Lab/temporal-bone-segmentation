# ABL Temporal Bone Segmentation Network

This repository contains the model and supporting files to build a Docker container for autonomous segmentation of the temporal bone, intended for use with [ABLInfer](https://github.com/Auditory-Biophysics-Lab/ablinfer).

This model is available on DockerHub as `uwoabl/tempseg`.

## Requirements
* Docker, preferably with GPU support (tested and working on Windows WSL2 + CUDA)
* CPU with AVX (Intel ix-2xxx [Sandy Bridge, 2011], AMD Bulldozer [late 2011] or better)
* At least 26GB *available* RAM/swap (works fine with 16GB RAM + additional swap)
* If using a GPU, at least 6GB VRAM (more is better)
