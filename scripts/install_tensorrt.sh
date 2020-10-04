#!/bin/bash

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804x86_64cuda-ubuntu1804.pin  
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

CUDA_VERSION="10.2.89"
TRT_VERSION="7.1.3-1+cuda10.2"
OS="ubuntu1804"

set -e

# purge existing CUDA first
if [ -e /usr/local/cuda ]; then
    read -p "Existing CUDA will be purged and reinstalled. Do you wish to proceed?" yn
    case $yn in
        [Yy]* ) sudo apt --purge remove "*cuda*"; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
fi

wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-repo-${OS}_${CUDA_VERSION}-1_amd64.deb
sudo dpkg -i cuda-repo-*.deb
wget https://developer.download.nvidia.com/compute/machine-learning/repos/${OS}/x86_64/nvidia-machine-learning-repo-${OS}_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-*.deb
sudo apt-get update

# install cuda, cudnn, and tensorrt
sudo apt-get install -y cuda
sudo apt-get install -y libcudnn7 libcudnn7-dev
sudo apt-get install libnvinfer7=${TRT_VERSION} libnvonnxparsers7=${TRT_VERSION} libnvparsers7=${TRT_VERSION} 
                     libnvinfer-plugin7=${TRT_VERSION} libnvinfer-dev=${TRT_VERSION} libnvonnxparsers-dev=${TRT_VERSION} \ 
                     libnvparsers-dev=${TRT_VERSION} libnvinfer-plugin-dev=${TRT_VERSION} python-libnvinfer=${TRT_VERSION} \
                     python3-libnvinfer=${TRT_VERSION} uff-converter-tf=${TRT_VERSION}
sudo apt-mark hold libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev \
                   libnvparsers-dev libnvinfer-plugin-dev python-libnvinfer python3-libnvinfer uff-converter-tf