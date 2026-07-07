# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
#
# Reproducible PyTorch + CUDA image for 3D Gaussian Splatting training,
# rendering, and metrics. The base image owns Python, torch, torchvision,
# CUDA runtime wheels, and CUDA development tooling.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel

FROM ${PYTORCH_IMAGE} AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ARG MAX_JOBS=8

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        ninja-build \
        cmake \
        git \
        nano \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip "setuptools<82" wheel

COPY requirements/constraints.txt \
     requirements/project.txt \
     requirements/docker-test.txt \
     /tmp/requirements/

RUN mkdir -p /opt/pip \
    && { \
        python -m pip freeze | grep -E '^(torch|torchvision|torchaudio|triton|pytorch-triton)=='; \
        cat /tmp/requirements/constraints.txt; \
    } > /opt/pip/constraints.txt

ENV PIP_CONSTRAINT=/opt/pip/constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        -r /tmp/requirements/project.txt \
        -r /tmp/requirements/docker-test.txt

FROM deps AS project

WORKDIR /opt/gaussian-splatting

COPY docker/patches/diff-gaussian-rasterization-cstdint.patch docker/patches/
COPY submodules/diff-gaussian-rasterization submodules/diff-gaussian-rasterization
COPY submodules/simple-knn submodules/simple-knn
COPY submodules/fused-ssim submodules/fused-ssim

RUN git apply --ignore-space-change \
        --directory=submodules/diff-gaussian-rasterization \
        docker/patches/diff-gaussian-rasterization-cstdint.patch

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization \
    && python -m pip install --no-build-isolation ./submodules/simple-knn \
    && python -m pip install --no-build-isolation ./submodules/fused-ssim

COPY arguments arguments
COPY gaussian_renderer gaussian_renderer
COPY lpipsPyTorch lpipsPyTorch
COPY scene scene
COPY utils utils
COPY train.py render.py metrics.py full_eval.py convert.py ./

COPY docker/entrypoint.sh /usr/local/bin/pytorch-cuda-entrypoint
COPY docker/runtime_info.py /usr/local/bin/pytorch-cuda-runtime-info
COPY docker/smoke_test.py /usr/local/bin/pytorch-cuda-smoke-test

RUN chmod +x \
        /usr/local/bin/pytorch-cuda-entrypoint \
        /usr/local/bin/pytorch-cuda-runtime-info \
        /usr/local/bin/pytorch-cuda-smoke-test \
    && mkdir -p /workspace/project /workspace/datasets /workspace/runs /workspace/artifacts /workspace/cache

ENV PYTHONPATH=/opt/gaussian-splatting \
    GS_DATASETS=/workspace/datasets \
    GS_RUNS=/workspace/runs \
    GS_ARTIFACTS=/workspace/artifacts \
    XDG_CACHE_HOME=/workspace/cache \
    TORCH_HOME=/workspace/cache/torch

WORKDIR /workspace/project

ENTRYPOINT ["pytorch-cuda-entrypoint"]
CMD ["bash"]
