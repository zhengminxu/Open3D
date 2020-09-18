FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Build system deps
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git

# Open3D default deps
# Equivalent to ./util/install_deps_ubuntu.sh
RUN apt-get update && apt-get install -y \
    xorg-dev \
    libglu1-mesa-dev \
    python3-dev \
    libsdl2-dev \
    libc++-7-dev \
    libc++abi-7-dev \
    clang \
    ninja-build \
    libxi-dev \
    libtbb-dev \
    libosmesa6-dev
RUN clang++ --version
# RUN g++ --version

# # Clone Open3D
# RUN git clone --recursive https://github.com/intel-isl/Open3D.git
# WORKDIR Open3D
# RUN ls -alh
# RUN git --no-pager log --name-status HEAD^..HEAD

# # Build Open3D
# RUN mkdir build
# WORKDIR build
# RUN cmake \
#     -DBUILD_BENCHMARKS=OFF \
#     -DBUILD_FILAMENT_FROM_SOURCE=ON \
#     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#     -DBUILD_SHARED_LIBS=OFF \
#     -DBUILD_GUI=ON \
#     -DBUILD_TENSORFLOW_OPS=OFF \
#     -DBUILD_PYTORCH_OPS=OFF \
#     -DBUILD_CUDA_MODULE=OFF \
#     -DCMAKE_C_COMPILER=gcc \
#     -DCMAKE_CXX_COMPILER=g++ \
#     -DBUILD_UNIT_TESTS=ON \
#     -DCMAKE_INSTALL_PREFIX=~/open3d_install ..
# RUN make -j$(nproc)
