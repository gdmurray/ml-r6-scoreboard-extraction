#!/usr/bin/sh

apt update
apt-get install -y --no-install-recommends apt-utils
apt-get install -y build-essential cmake unzip pkg-config
apt-get install -y libjpeg-dev libpng-dev libtiff-dev
add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
apt install -y libjasper1 libjasper-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev
apt-get install libgtk-3-dev
apt-get install libatlas-base-dev gfortran
apt-get install python3.6-dev
wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/4.1.2.tar.gz
wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/4.1.2.tar.gz
tar -xvzf opencv.tar.gz -C opencv
tar -xvzf opencv_contrib.tar.gz -C opencv_contrib
apt-get install -y python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=ON