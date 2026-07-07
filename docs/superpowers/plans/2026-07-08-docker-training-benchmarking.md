# Docker Training And Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Docker workflow for training and benchmarking this 3D Gaussian Splatting project with PyTorch 2.12.1, Python 3.12, and CUDA 12.6 / 13.0 / 13.2 image lanes.

**Architecture:** Keep the original optimizer and benchmarking scripts intact. Build a container image that copies the project source, installs runtime dependencies, compiles the three CUDA extension submodules as non-editable packages, and exposes stable dataset, run, artifact, and cache mount points. Add helper scripts and documentation so users can verify image readiness before running dataset-backed training or benchmark jobs.

**Tech Stack:** Docker, Docker Compose, official `pytorch/pytorch` CUDA devel images, Python 3.12, PyTorch 2.12.1, CUDA extension builds via `torch.utils.cpp_extension`, Bash entrypoint, Python smoke helpers, Markdown documentation.

## Global Constraints

- Default image lane: `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`.
- Supported build-arg image lanes: `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel`, `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`, and `pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel`.
- Default `TORCH_CUDA_ARCH_LIST`: `8.6;8.9;12.0`.
- Default `MAX_JOBS`: `8`.
- Runtime working directory: `/workspace/project`.
- Copied project source directory: `/opt/gaussian-splatting`.
- Runtime mount directories: `/workspace/datasets`, `/workspace/runs`, `/workspace/artifacts`, and `/workspace/cache`.
- Do not modify training math, rasterizer behavior, optimizer defaults, or dataset parsing.
- Install CUDA submodules during image build as non-editable packages with `pip install --no-build-isolation`.
- Do not pin or override torch packages supplied by the PyTorch base image.

---

## File Structure

- Create `requirements/project.txt`: Python runtime dependencies not owned by the PyTorch image.
- Create `requirements/docker-test.txt`: intentionally minimal smoke-test dependency input.
- Create `requirements/constraints.txt`: non-torch dependency bounds and build tooling bounds.
- Create `docker/entrypoint.sh`: runtime directory creation and command handoff.
- Create `docker/runtime_info.py`: environment and extension-readiness report.
- Create `docker/smoke_test.py`: no-dataset CUDA and extension smoke test.
- Create `Dockerfile`: multi-stage PyTorch CUDA image that installs dependencies and compiles submodules.
- Create `docker-compose.yml`: convenience service with GPU, build args, and host path mounts.
- Create `.dockerignore`: keep build context focused and avoid sending datasets, runs, caches, and Git metadata.
- Create `docs/docker-training-benchmarking.md`: project overview, readiness matrix, build commands, run commands, and verification notes.
- Modify `docs/docker-training-benchmarking.md` after verification if Docker build or smoke checks reveal exact blockers.

---

### Task 1: Runtime Dependencies And Helper Scripts

**Files:**
- Create: `requirements/project.txt`
- Create: `requirements/docker-test.txt`
- Create: `requirements/constraints.txt`
- Create: `docker/entrypoint.sh`
- Create: `docker/runtime_info.py`
- Create: `docker/smoke_test.py`

**Interfaces:**
- Produces command `pytorch-cuda-entrypoint`, copied from `docker/entrypoint.sh`.
- Produces command `pytorch-cuda-runtime-info`, copied from `docker/runtime_info.py`.
- Produces command `pytorch-cuda-smoke-test`, copied from `docker/smoke_test.py`.
- Later Dockerfile task consumes the three requirements files and three helper scripts.

- [ ] **Step 1: Create runtime helper directory and requirements directory**

Use `apply_patch` to add files directly. No shell directory creation is required because `apply_patch` creates parent directories for new files in this workspace.

- [ ] **Step 2: Add `requirements/project.txt`**

Write this exact file:

```text
numpy>=1.26,<3
pillow>=10,<13
opencv-python-headless>=4.9,<5
plyfile>=0.8.1,<2
tqdm>=4.66,<5
joblib>=1.3,<2
tensorboard>=2.16,<3
```

- [ ] **Step 3: Add `requirements/docker-test.txt`**

Write this exact file:

```text
# Reserved for smoke-test-only Python dependencies.
# Keep this file minimal; current smoke helpers use only project/runtime deps.
```

- [ ] **Step 4: Add `requirements/constraints.txt`**

Write this exact file:

```text
setuptools<82
wheel>=0.43,<1
numpy>=1.26,<3
pillow>=10,<13
opencv-python-headless>=4.9,<5
plyfile>=0.8.1,<2
tqdm>=4.66,<5
joblib>=1.3,<2
tensorboard>=2.16,<3
```

- [ ] **Step 5: Add `docker/entrypoint.sh`**

Write this exact file:

```bash
#!/usr/bin/env bash
set -euo pipefail

export GS_DATASETS="${GS_DATASETS:-/workspace/datasets}"
export GS_RUNS="${GS_RUNS:-/workspace/runs}"
export GS_ARTIFACTS="${GS_ARTIFACTS:-/workspace/artifacts}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"

mkdir -p \
  "${GS_DATASETS}" \
  "${GS_RUNS}" \
  "${GS_ARTIFACTS}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}"

exec "$@"
```

- [ ] **Step 6: Add `docker/runtime_info.py`**

Write this exact file:

```python
#!/usr/bin/env python
from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImportCheck:
    name: str
    ok: bool
    detail: str


def check_import(module_name: str, attr_name: str | None = None) -> ImportCheck:
    try:
        module = importlib.import_module(module_name)
        if attr_name is not None:
            getattr(module, attr_name)
        detail = "ok"
        version = getattr(module, "__version__", None)
        if version:
            detail = f"ok version={version}"
        return ImportCheck(module_name if attr_name is None else f"{module_name}.{attr_name}", True, detail)
    except Exception as exc:
        return ImportCheck(module_name if attr_name is None else f"{module_name}.{attr_name}", False, f"{type(exc).__name__}: {exc}")


def print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def main() -> int:
    print_kv("python", sys.version.replace("\n", " "))
    print_kv("platform", platform.platform())
    print_kv("torch", torch.__version__)
    print_kv("torch_cuda", torch.version.cuda)
    print_kv("cudnn", torch.backends.cudnn.version())
    print_kv("cuda_available", torch.cuda.is_available())

    if torch.cuda.is_available():
        print_kv("cuda_device_count", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print_kv(f"cuda_device_{index}", f"{props.name}, capability={props.major}.{props.minor}, memory={props.total_memory}")

    for key in ("PYTHONPATH", "TORCH_CUDA_ARCH_LIST", "GS_DATASETS", "GS_RUNS", "GS_ARTIFACTS", "XDG_CACHE_HOME", "TORCH_HOME"):
        print_kv(key.lower(), os.environ.get(key, ""))

    checks = (
        check_import("diff_gaussian_rasterization", "GaussianRasterizer"),
        check_import("diff_gaussian_rasterization", "SparseGaussianAdam"),
        check_import("simple_knn._C", "distCUDA2"),
        check_import("fused_ssim", "fused_ssim"),
    )
    failed = False
    for check in checks:
        status = "ok" if check.ok else "fail"
        print_kv(f"import_{check.name}", f"{status}: {check.detail}")
        failed = failed or not check.ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Add `docker/smoke_test.py`**

Write this exact file:

```python
#!/usr/bin/env python
from __future__ import annotations

import importlib

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    result = tensor.square().sum().item()
    if result != 14.0:
        raise RuntimeError(f"unexpected CUDA tensor result: {result}")
    print(f"torch cuda ok: torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")


def check_simple_knn() -> None:
    from simple_knn._C import distCUDA2

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    distances = distCUDA2(points)
    if distances.shape[0] != points.shape[0]:
        raise RuntimeError(f"unexpected simple_knn output shape: {tuple(distances.shape)}")
    if not torch.isfinite(distances).all():
        raise RuntimeError("simple_knn returned non-finite distances")
    print("simple_knn ok")


def check_fused_ssim() -> None:
    from fused_ssim import fused_ssim

    first = torch.rand((1, 3, 16, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    second = first.detach().clone()
    value = fused_ssim(first, second)
    if not torch.isfinite(value):
        raise RuntimeError("fused_ssim returned a non-finite value")
    value.backward()
    if first.grad is None:
        raise RuntimeError("fused_ssim did not populate gradients")
    print(f"fused_ssim ok: value={value.detach().item():.6f}")


def check_rasterizer_imports() -> None:
    module = importlib.import_module("diff_gaussian_rasterization")
    for attr_name in ("GaussianRasterizationSettings", "GaussianRasterizer", "SparseGaussianAdam"):
        getattr(module, attr_name)
    print("diff_gaussian_rasterization imports ok")


def main() -> int:
    require_cuda()
    check_rasterizer_imports()
    check_simple_knn()
    check_fused_ssim()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run local Python syntax checks**

Run:

```bash
python -m py_compile docker/runtime_info.py docker/smoke_test.py
```

Expected output:

```text
```

The command should exit `0` with no output.

- [ ] **Step 9: Commit Task 1**

Run:

```bash
git add requirements/project.txt requirements/docker-test.txt requirements/constraints.txt docker/entrypoint.sh docker/runtime_info.py docker/smoke_test.py
git commit -m "build: add docker runtime helpers"
```

Expected output includes:

```text
build: add docker runtime helpers
```

---

### Task 2: Dockerfile, Compose, And Build Context

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

**Interfaces:**
- Consumes `requirements/project.txt`, `requirements/docker-test.txt`, `requirements/constraints.txt`, `docker/entrypoint.sh`, `docker/runtime_info.py`, and `docker/smoke_test.py`.
- Produces image commands `pytorch-cuda-runtime-info` and `pytorch-cuda-smoke-test`.
- Produces Compose service `gaussian-splatting`.

- [ ] **Step 1: Add `.dockerignore`**

Write this exact file:

```text
.git
.gitignore
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
build/
dist/
*.egg-info/
data/
datasets/
runs/
output/
eval/
artifacts/
cache/
*.zip
*.tar
*.tar.gz
*.7z
```

- [ ] **Step 2: Add `Dockerfile`**

Write this exact file:

```dockerfile
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

RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

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

RUN python -m pip install --no-cache-dir \
        -r /tmp/requirements/project.txt \
        -r /tmp/requirements/docker-test.txt

FROM deps AS project

COPY . /opt/gaussian-splatting

WORKDIR /opt/gaussian-splatting

RUN python -m pip install --no-cache-dir --no-build-isolation ./submodules/diff-gaussian-rasterization \
    && python -m pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn \
    && python -m pip install --no-cache-dir --no-build-isolation ./submodules/fused-ssim

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
```

- [ ] **Step 3: Add `docker-compose.yml`**

Write this exact file:

```yaml
services:
  gaussian-splatting:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        PYTORCH_IMAGE: ${PYTORCH_IMAGE:-pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel}
        TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-8.6;8.9;12.0}
        MAX_JOBS: ${MAX_JOBS:-8}
    image: ${GS_IMAGE:-gaussian-splatting:pytorch-2.12.1-cuda13.0}
    gpus: all
    shm_size: ${GS_SHM_SIZE:-16gb}
    working_dir: /workspace/project
    environment:
      GS_DATASETS: /workspace/datasets
      GS_RUNS: /workspace/runs
      GS_ARTIFACTS: /workspace/artifacts
      XDG_CACHE_HOME: /workspace/cache
      TORCH_HOME: /workspace/cache/torch
      NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-all}
      NVIDIA_DRIVER_CAPABILITIES: ${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
    volumes:
      - ${GS_DATASETS_HOST:-./data}:/workspace/datasets
      - ${GS_RUNS_HOST:-./runs}:/workspace/runs
      - ${GS_ARTIFACTS_HOST:-./artifacts}:/workspace/artifacts
      - ${GS_CACHE_HOST:-./cache}:/workspace/cache
    command: bash
```

- [ ] **Step 4: Run Docker Compose config validation**

Run:

```bash
docker compose config
```

Expected output contains:

```text
gaussian-splatting:
```

- [ ] **Step 5: Run Dockerfile text sanity checks**

Run:

```bash
python -c "from pathlib import Path; p=Path('Dockerfile').read_text(); assert 'pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel' in p; assert '--no-build-isolation ./submodules/diff-gaussian-rasterization' in p; assert 'ENTRYPOINT [\"pytorch-cuda-entrypoint\"]' in p"
```

Expected output:

```text
```

The command should exit `0` with no output.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "build: add docker image and compose"
```

Expected output includes:

```text
build: add docker image and compose
```

---

### Task 3: Project Overview And Docker Workflow Documentation

**Files:**
- Create: `docs/docker-training-benchmarking.md`

**Interfaces:**
- Consumes Docker assets and helper command names from Tasks 1 and 2.
- Produces the user-facing reference for build, smoke, training, rendering, metrics, full evaluation, paths, and readiness states.

- [ ] **Step 1: Add `docs/docker-training-benchmarking.md`**

Write a Markdown document with these exact top-level headings:

```markdown
# Docker Training And Benchmarking

## Project Overview

## Runtime Components

## Dataset And Output Layout

## Docker Image Lanes

## Build Commands

## Runtime Checks

## Training Workflows

## Rendering And Metrics

## Full Benchmark Workflow

## Compose Reference

## Readiness Matrix

## Known Risks And Readiness Notes
```

- [ ] **Step 2: Fill `Project Overview` and `Runtime Components`**

Include these facts:

```markdown
- `train.py` optimizes 3D Gaussians from COLMAP or NeRF synthetic inputs.
- `render.py` renders train/test image sets from a saved model.
- `metrics.py` computes SSIM, PSNR, and LPIPS from rendered outputs.
- `full_eval.py` orchestrates the paper-style train, render, and metrics workflow.
- `convert.py` prepares COLMAP datasets from input images when COLMAP and ImageMagick are available.
- `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, and `submodules/fused-ssim` are CUDA extensions compiled during Docker image build.
```

- [ ] **Step 3: Fill `Dataset And Output Layout`**

Include this path table:

```markdown
| Container Path | Purpose |
| --- | --- |
| `/workspace/datasets` | Mounted COLMAP, NeRF synthetic, or benchmark datasets |
| `/workspace/runs` | Training model outputs, checkpoints, TensorBoard event files |
| `/workspace/artifacts` | Rendered benchmark artifacts, copied reports, exported summaries |
| `/workspace/cache` | Torch, torchvision, LPIPS, pip, and model weight cache |
| `/opt/gaussian-splatting` | Copied project source and installed CUDA extension sources |
| `/workspace/project` | Default command working directory |
```

- [ ] **Step 4: Fill `Docker Image Lanes` and `Build Commands`**

Include these commands:

```bash
docker compose build
```

```bash
$env:PYTORCH_IMAGE = 'pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel'
docker compose build
```

```bash
$env:PYTORCH_IMAGE = 'pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel'
docker compose build
```

Also include the non-Compose build command:

```bash
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel -t gaussian-splatting:pytorch-2.12.1-cuda13.0 .
```

- [ ] **Step 5: Fill `Runtime Checks`**

Include these commands:

```bash
docker compose run --rm gaussian-splatting pytorch-cuda-runtime-info
```

```bash
docker compose run --rm gaussian-splatting pytorch-cuda-smoke-test
```

Explain that passing `pytorch-cuda-smoke-test` moves a lane from `build-verified` to `runtime-verified`.

- [ ] **Step 6: Fill `Training Workflows`**

Include these commands:

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden --disable_viewer
```

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-eval --eval --disable_viewer
```

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-fast --eval --optimizer_type sparse_adam --disable_viewer
```

- [ ] **Step 7: Fill `Rendering And Metrics`**

Include these commands:

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/render.py -s /workspace/datasets/garden -m /workspace/runs/garden-eval --skip_train
```

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/metrics.py -m /workspace/runs/garden-eval
```

Explain that metrics may download torchvision and LPIPS weights into `/workspace/cache` on first run.

- [ ] **Step 8: Fill `Full Benchmark Workflow`, `Compose Reference`, `Readiness Matrix`, and `Known Risks And Readiness Notes`**

Include this readiness table:

```markdown
| Lane | Current State | Verification Required |
| --- | --- | --- |
| PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | documented | Build image, run runtime info, run smoke test |
| PyTorch 2.12.1 + CUDA 12.6 + Python 3.12 | documented | Build image, run runtime info, run smoke test |
| PyTorch 2.12.1 + CUDA 13.2 + Python 3.12 | documented | Build image, run runtime info, run smoke test |
```

Include these known risks:

```markdown
- CUDA 13.x extension compilation is not claimed verified until the Docker build succeeds.
- `full_eval.py` uses string-based `os.system` calls and is less robust than direct command invocation.
- `full_eval.py` can write `timing.txt` from uninitialized timing variables in some skip-only modes.
- Full GPU determinism is not guaranteed by the existing optimizer.
- Offline metrics runs require a populated `/workspace/cache`.
```

- [ ] **Step 9: Run Markdown sanity checks**

Run:

```bash
python -c "from pathlib import Path; p=Path('docs/docker-training-benchmarking.md').read_text(); required=['# Docker Training And Benchmarking','## Runtime Checks','pytorch-cuda-smoke-test','pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel']; missing=[x for x in required if x not in p]; assert not missing, missing"
```

Expected output:

```text
```

The command should exit `0` with no output.

- [ ] **Step 10: Commit Task 3**

Run:

```bash
git add docs/docker-training-benchmarking.md
git commit -m "docs: add docker training workflow"
```

Expected output includes:

```text
docs: add docker training workflow
```

---

### Task 4: Verification And Readiness Update

**Files:**
- Modify: `docs/docker-training-benchmarking.md`

**Interfaces:**
- Consumes Docker assets from Tasks 1 and 2.
- Produces documented verification result for attempted Docker build and smoke commands.

- [ ] **Step 1: Check Docker client availability**

Run:

```bash
docker --version
docker compose version
```

Expected output contains:

```text
Docker version
Docker Compose version
```

- [ ] **Step 2: Build the default CUDA 13.0 image**

Run:

```bash
docker compose build
```

Expected output contains either successful image build completion or an exact error message to record in the docs.

- [ ] **Step 3: Run runtime info if build succeeds**

Run:

```bash
docker compose run --rm gaussian-splatting pytorch-cuda-runtime-info
```

Expected output contains:

```text
torch:
cuda_available:
import_diff_gaussian_rasterization.GaussianRasterizer:
```

- [ ] **Step 4: Run no-dataset smoke test if runtime info succeeds**

Run:

```bash
docker compose run --rm gaussian-splatting pytorch-cuda-smoke-test
```

Expected output contains:

```text
smoke test passed
```

- [ ] **Step 5: Update readiness section with observed result**

If build and smoke pass, update the CUDA 13.0 row in `docs/docker-training-benchmarking.md` from:

```markdown
| PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | documented | Build image, run runtime info, run smoke test |
```

to:

```markdown
| PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | runtime-verified | `docker compose build`, `pytorch-cuda-runtime-info`, and `pytorch-cuda-smoke-test` passed |
```

If build or smoke fails, update the same row to:

```markdown
| PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | blocked | See verification notes below for the exact failing command and error |
```

Then add a `## Verification Notes` section with the exact command and concise failure or success result.

- [ ] **Step 6: Run final static checks**

Run:

```bash
python -m py_compile docker/runtime_info.py docker/smoke_test.py
python -c "from pathlib import Path; files=['Dockerfile','docker-compose.yml','docs/docker-training-benchmarking.md']; [Path(f).read_text() for f in files]"
docker compose config
```

Expected results:

```text
```

The Python commands should exit `0` with no output. `docker compose config` should print a resolved Compose configuration containing `gaussian-splatting`.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add docs/docker-training-benchmarking.md
git commit -m "docs: record docker verification status"
```

Expected output includes either:

```text
docs: record docker verification status
```

or:

```text
nothing to commit, working tree clean
```

Use the second outcome only if Task 4 did not change the docs because no Docker build was attempted.

---

## Self-Review

Spec coverage:

- Requirements files are covered by Task 1.
- Runtime helper scripts are covered by Task 1.
- Dockerfile, non-editable CUDA submodule installs, default CUDA 13.0 lane, and build args are covered by Task 2.
- Docker Compose GPU service, mounts, and build args are covered by Task 2.
- Project overview, workflows, readiness matrix, and known risks are covered by Task 3.
- Build and smoke verification plus readiness updates are covered by Task 4.

Placeholder scan:

- The plan contains no deferred implementation placeholders.

Type and interface consistency:

- Helper script command names match Dockerfile copies and documentation commands.
- Compose service name is consistently `gaussian-splatting`.
- Runtime paths match the spec: `/workspace/project`, `/opt/gaussian-splatting`, `/workspace/datasets`, `/workspace/runs`, `/workspace/artifacts`, and `/workspace/cache`.
