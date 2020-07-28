## STEP 1: BUILD CUDARESIZE.SO

## This should be the same image as the one used at the end so the CUDA versions match
FROM tensorflow/tensorflow:1.15.2-gpu-py3

## Install git
RUN apt-get update
RUN apt-get install -y git

## Copy in the CUDA resize code
COPY cudaresize /tmp/cudaresize

## Clone the repository
WORKDIR /tmp/cudaresize
RUN git clone https://github.com/DannyRuijters/CubicInterpolationCUDA

## Now checkout the right version
WORKDIR /tmp/cudaresize/CubicInterpolationCUDA
RUN git checkout d52d23cf4ad152d011101b19d305d64540b665fd

## Apply the patch for modern CUDA; this fixes compilation and removes the OpenGL dependency
RUN git apply ../moderncuda.patch

## Lastly, compile the library
WORKDIR /tmp/cudaresize
ENV CUDA_PATH /usr/local/cuda
RUN make

## STEP 2: BUILD THE IMAGE

## Base on the most recent version of TensorFlor 1.15 with Python 3 (MUST BE 1.15.x!)
FROM tensorflow/tensorflow:1.15.2-gpu-py3

## Install dependencies
RUN python3 -m pip --no-cache-dir install SimpleITK
RUN python3 -m pip --no-cache-dir install niftynet

## Add the network and handle script
COPY niftynet /var/niftynet
COPY --from=0 /tmp/cudaresize/libcudaresize.so /var/niftynet/
RUN mkdir /var/niftynet/work
RUN chmod -R g+rX,o+rX /var/niftynet
RUN chmod g+rwX,o+rwX /var/niftynet/work

ENTRYPOINT ["/var/niftynet/handle.py"]
