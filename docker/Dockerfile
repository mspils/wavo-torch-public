# syntax = docker/dockerfile:1.3
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
RUN apt update && apt install -y \
    pip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /home/wavo-torch/notebooks
CMD jupyter-lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root
#--notebook-dir=/tf 
LABEL name="kiwavo-tk" version="0.0.4" maintainer="Michel Spils <msp@informatik.uni-kiel.de>"
# docker run -u $(id -u):$(id -g) --gpus all --rm -it -p 9999:9999 --name wln --workdir /../app/ki-wavo/notebooks 
# --mount type=bind,source="$(pwd)",target=/app  kiwavo/notebook:latest jupyter-lab --ip 0.0.0.0 --port 9999
#pytorch-forecasting==1.0.0
