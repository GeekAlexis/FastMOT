# Jetpack 4.4 (OpenCV, CUDA, TensorRT) is required before running this script
DIR=$HOME
BASEDIR=$(dirname "$0")

# export CUDA paths
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}' >> ~/.bashrc 
source ~/.bashrc

# install python dependencies
sudo apt-get update
sudo apt-get install python3-pip libhdf5-serial-dev hdf5-tools
sudo pip3 install cython numpy scipy pycuda mpipe sharedmem
sudo pip3 install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.2+nv20.4

# prepare INT8 calibration data
source $BASEDIR/analytics/models/prepare_calib_data.sh $BASEDIR/analytics/models

# install ray
# sudo apt-get install build-essential openjdk-8-jdk unzip
# export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-arm64"
# wget https://github.com/bazelbuild/bazel/releases/download/1.0.0/bazel-1.0.0-dist.zip -P $DIR
# unzip $DIR/bazel-1.0.0-dist.zip -d bazel-1.0.0
# $DIR/bazel-1.0.0/compile.sh
# mkdir $HOME/.bazel/bin
# mv $DIR/bazel-1.0.0/output/bazel $HOME/.bazel/bin/

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# source $CARGO_HOME/.cargo/env
# cargo install py-spy

# git clone --branch releases/0.8.3 https://github.com/ray-project/ray.git $DIR
# # git checkout fae99ecb8e8d750bddcb3674f720f068541dc15d
# cd $DIR/ray/thirdparty/patches
# wget https://gist.githubusercontent.com/heavyinfo/25cf56fe0b5f8509dd0120257d008d3f/raw/351d8aa050c1c957bb10d9f5671ea44130de6211/rules_boost-context-thread-arm64.patch
# patch $DIR/ray/bazel/ray_deps_setup.bzl $BASEDIR/ray.patch

# cd $DIR/ray/python
# sudo python3 setup.py bdist_wheel
# cd $DIR/ray/python/dist
# pip3 install ray-0.8.3-cp36-cp36m-linux_aarch64.whl
# # remove py-spy requirement from ray/python/setup.py if py-spy not found