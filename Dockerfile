FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM nvcr.io/nvidia/pytorch:22.09-py3

# Set non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update 
RUN apt-get install -y tmux htop

# Install Open3d for PatchworkPP
RUN pip install open3d 
RUN apt-get install -y libeigen3-dev

# Install spareseconv & auxiliary libs
RUN pip install imageio
RUN pip install tensorflow
RUN pip install protobuf==3.20.1
