#!/bin/bash

OPENCV_VERSION="4.1.1" 
ARCH_BIN=7.5 # compute capabilities can be found here https://developer.nvidia.com/cuda-gpus#compute
DIR=$HOME

set -e

# install dependencies:
sudo apt-get -y update
sudo apt-get install -y build-essential \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-venv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libdc1394-22-dev \
    libavresample-dev

cd $DIR
wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
mv opencv-${OPENCV_VERSION} OpenCV

wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip
mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
mv opencv_contrib OpenCV

cd OpenCV
python3 -m venv opencv4
source opencv4/bin/activate
pip install wheel
pip install numpy

mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH='../opencv_contrib/modules ..' \
    -D PYTHON_EXECUTABLE='../opencv4/bin/python' \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=${ARCH_BIN} \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    ../

make -j8
sudo make install
sudo ldconfig