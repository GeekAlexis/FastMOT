ARG TRT_IMAGE_VERSION=20.09
FROM nvcr.io/nvidia/tensorrt:${TRT_IMAGE_VERSION}-py3

ARG OPENCV_VERSION=4.1.1
ARG APP_DIR=/usr/src/app
ARG SCRIPT_DIR=/opt/tensorrt/python
ENV HOME ${APP_DIR}

# install OpenCV and FastMOT dependencies
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    wget unzip \
    build-essential cmake pkg-config \
    libgtk-3-dev libcanberra-gtk3-module \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    gfortran libatlas-base-dev \
    python3-dev \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libtbb2 libtbb-dev libdc1394-22-dev && \
    pip install --no-cache-dir numpy==1.18.0

# build OpenCV
WORKDIR ${HOME}
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv_contrib-${OPENCV_VERSION} OpenCV/opencv_contrib

WORKDIR ${HOME}/OpenCV/build
RUN cmake \ 
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=${HOME}/OpenCV/opencv_contrib/modules \
    -DINSTALL_PYTHON_EXAMPLES=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_PROTOBUF=OFF \
    -DENABLE_FAST_MATH=ON \
    -DWITH_TBB=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_CUDA=OFF \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_FFMPEG=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf ${HOME}/OpenCV && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove

# install Python dependencies
WORKDIR ${APP_DIR}/FastMOT
COPY . .
RUN dpkg -i ${SCRIPT_DIR}/*-tf_*.deb && \
    pip install --no-cache-dir cython && \
    pip install --no-cache-dir -r requirements.txt
