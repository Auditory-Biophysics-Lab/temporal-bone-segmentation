# ABL Temporal Bone Segmentation Network

This repository contains the model and supporting files to build a Docker container for autonomous segmentation of the temporal bone, intended for use with [ABLInfer](https://github.com/Auditory-Biophysics-Lab/ablinfer).

This model is available on DockerHub as `uwoabl/temporal-bone-segmentation`.

## Requirements
* Docker, preferably with GPU support (tested and working on Windows WSL2 + CUDA)
* CPU with AVX (Intel ix-2xxx [Sandy Bridge, 2011], AMD Bulldozer [late 2011] or better)
* At least 26GB *available* RAM/swap (works fine with 16GB RAM + additional swap)
* If using a GPU, at least 6GB VRAM (more is better)

## Building the Image
Building the image is generally a simple affair: change into the directory and run `docker build -t <image-name> .`. Docker tends to make a bit of a mess with intermediate images; this can speed up future builds, but can also break things if the intermediate image that ran `apt-get update` is out of date, since the build will try to retrieve non-existant packages. In this case, running `docker build --no-cache -t <image-name> .` will solve the issue.
