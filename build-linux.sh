#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin"
export CUDACXX="/usr/local/cuda-12.8/bin/nvcc"
export CUDA_HOME="/usr/local/cuda-12.8"

SRC_DIR="/mnt/c/Users/A-511/OneDrive/Desktop/projects/Strata"
BUILD_DIR="${SRC_DIR}/build-linux"

mkdir -p "${BUILD_DIR}"
cd "${SRC_DIR}"

cmake -B "${BUILD_DIR}" -S "${SRC_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="/usr/local/cuda-12.8/bin/nvcc" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DSTRATA_PORTABLE=ON

ninja -C "${BUILD_DIR}" strata -j 16
