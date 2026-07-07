# Docker Training And Benchmarking Design

## Context

This repository is a GraphDECO-style 3D Gaussian Splatting checkout with Python
training, rendering, metrics, and COLMAP conversion entry points. The local tree
contains three CUDA extension packages under `submodules/`:

- `diff-gaussian-rasterization`, including the accelerated rasterizer branch
  with antialiasing and sparse Adam support.
- `simple-knn`, used during Gaussian initialization and densification.
- `fused-ssim`, used as an optional faster SSIM loss during training.

The checked-in `env.yml` targets Python 3.10, PyTorch 2.1.2, and CUDA 11.8.
The requested Docker target is newer: PyTorch 2.12.1, Python 3.12 from the
official PyTorch image, and CUDA 12.6 / 13.0 / 13.2 image lanes. Docker Hub
currently publishes official `pytorch/pytorch` devel tags for these lanes:

- `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel`
- `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`
- `pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel`

The original upstream project documents CUDA-ready GPUs with compute capability
7.0+, CUDA SDK 11 for the classic setup, and notes that newer Python, PyTorch,
and CUDA environments can work when PyTorch's CUDA runtime and the installed
CUDA SDK match. Upstream issues and pull requests also show CUDA 12 and Docker
work is active but not fully settled. The Docker integration must therefore
make compatibility evidence explicit rather than imply untested support.

## Goal

Provide a reproducible Docker workflow for training and benchmarking this
project without restructuring the original optimizer internals. The container
should build the CUDA extensions during image creation, expose stable host
mount points for datasets, logs, and artifacts, and provide repeatable runtime
checks for PyTorch, CUDA, and extension readiness.

## Non-Goals

- Do not package or build the SIBR viewers in the training Docker image.
- Do not convert the project into a full Python package unless required for
  Docker correctness.
- Do not modify training math, rasterizer behavior, optimizer defaults, or
  dataset parsing as part of the Docker lane.
- Do not require editable installs for normal container use.

## Recommended Approach

Use a container-first training lane around the existing source tree. The image
copies the project into `/opt/gaussian-splatting`, installs dependency wheels,
then installs CUDA submodules as normal non-editable packages during the image
build. Runtime commands execute from `/workspace/project` with the copied source
available on `PYTHONPATH`.

This favors reproducibility over live-edit convenience: the built image captures
the Python dependencies, CUDA extension build output, and the project revision
used for training or benchmarking.

## Docker Image Design

The default Docker image should use:

- `ARG PYTORCH_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`
- `ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"`
- `ARG MAX_JOBS=8`

The default lane is CUDA 13.0 because it is one of the requested targets. The
same Dockerfile must allow CUDA 12.6 and CUDA 13.2 through `PYTORCH_IMAGE`.

The image should:

- inherit Python, torch, torchvision, CUDA runtime wheels, and CUDA development
  tooling from the official PyTorch image;
- install OS packages needed for extension builds and headless image handling:
  compiler toolchain, Ninja, CMake, Git, OpenGL/EGL runtime libraries, GLib,
  OpenMP runtime, X11 compatibility libraries, and CA certificates;
- upgrade pip and keep `setuptools<82` to avoid churn in legacy `setup.py`
  extension builds;
- create a generated constraints file that freezes torch, torchvision,
  torchaudio, triton, and pytorch-triton packages already present in the base
  image;
- install project dependencies from repository-owned requirements files;
- copy the repository into the image;
- install `submodules/diff-gaussian-rasterization`,
  `submodules/simple-knn`, and `submodules/fused-ssim` with
  `pip install --no-build-isolation`;
- set runtime cache and artifact defaults under `/workspace`.

## Dependency Files

Add a `requirements/` directory with these files:

- `requirements/project.txt`: runtime Python packages not owned by the PyTorch
  base image. Expected packages are `numpy`, `pillow`, `opencv-python-headless`,
  `plyfile`, `tqdm`, `joblib`, and `tensorboard`.
- `requirements/docker-test.txt`: small dependencies needed only by smoke or
  validation helpers. This should stay empty or minimal unless a helper truly
  requires an external package.
- `requirements/constraints.txt`: non-torch package version bounds and build
  tooling bounds. Torch packages should not be pinned here because the base
  image owns the torch lane.

Python 3.12 compatibility should be handled by choosing packages with current
Python 3.12 wheels. If any dependency lacks a wheel for the selected image, the
Docker build should fail early during dependency installation rather than at
training time.

## Runtime Layout

The container should use stable paths:

- `/workspace/project`: working directory for commands.
- `/opt/gaussian-splatting`: immutable copied project source.
- `/workspace/datasets`: mounted datasets.
- `/workspace/runs`: training outputs and TensorBoard logs.
- `/workspace/artifacts`: benchmark summaries, rendered outputs, exported
  metrics, and copied reports.
- `/workspace/cache`: torch, torchvision, pip, and optional model-weight cache.

The image should set:

- `PYTHONPATH=/opt/gaussian-splatting`
- `TORCH_HOME=/workspace/cache/torch`
- `XDG_CACHE_HOME=/workspace/cache`
- `GS_DATASETS=/workspace/datasets`
- `GS_RUNS=/workspace/runs`
- `GS_ARTIFACTS=/workspace/artifacts`

The training scripts already support explicit `-s/--source_path` and
`-m/--model_path`, so Docker does not need to change the optimizer for path
control.

## Runtime Helper Scripts

Add helper scripts under `docker/` and copy them to `/usr/local/bin`:

- `docker/entrypoint.sh`: creates runtime directories and then executes the
  requested command.
- `docker/runtime_info.py`: prints Python version, torch version, CUDA version,
  cuDNN version, GPU visibility, selected environment variables, and extension
  import status.
- `docker/smoke_test.py`: verifies torch CUDA availability, imports all three
  CUDA extensions, runs a tiny `simple_knn.distCUDA2` call, runs a tiny
  `fused_ssim` call, and reports whether sparse Adam is importable from the
  rasterizer package.

The smoke test should avoid requiring a full dataset. A dataset-backed training
smoke test belongs in documentation and can be run when a small COLMAP or NeRF
synthetic fixture is mounted.

## Docker Compose Design

Add `docker-compose.yml` for convenience. It should:

- build from the repository `Dockerfile`;
- pass `PYTORCH_IMAGE`, `TORCH_CUDA_ARCH_LIST`, and `MAX_JOBS` build args from
  environment variables with documented defaults;
- request all NVIDIA GPUs through Compose GPU support;
- mount datasets, runs, artifacts, and cache from host-controlled paths;
- keep the default command as `bash`;
- expose port `6009` only for the optional network viewer path.

The compose file should not hard-code host dataset locations. It should allow
environment overrides such as:

- `GS_DATASETS_HOST=./data`
- `GS_RUNS_HOST=./runs`
- `GS_ARTIFACTS_HOST=./artifacts`
- `GS_CACHE_HOST=./cache`

## Documented Workflows

The documentation should cover:

1. Build the default CUDA 13.0 lane.
2. Build CUDA 12.6 and CUDA 13.2 lanes with `PYTORCH_IMAGE`.
3. Run runtime info.
4. Run the no-dataset smoke test.
5. Train a mounted COLMAP dataset with explicit `-s` and `-m`.
6. Train with evaluation split and disabled network viewer.
7. Render a trained model.
8. Compute metrics.
9. Run full evaluation when the official benchmark datasets are mounted.
10. Collect outputs from `/workspace/runs` and `/workspace/artifacts`.

The docs should explain that LPIPS metrics may download torchvision and LPIPS
weights unless they are already present under the mounted torch cache. For
network-restricted benchmark runs, users should pre-populate or persist
`/workspace/cache`.

## Readiness Matrix

The documentation should classify support in three states:

- `documented`: the lane is described and the image tag exists.
- `build-verified`: the image builds and all CUDA extensions compile.
- `runtime-verified`: `runtime_info.py` and `smoke_test.py` pass with a visible
  NVIDIA GPU.

Initial expected readiness:

| Lane | Initial State | Main Risk |
| --- | --- | --- |
| PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | documented | CUDA extension compile compatibility |
| PyTorch 2.12.1 + CUDA 12.6 + Python 3.12 | documented | CUDA extension compile compatibility |
| PyTorch 2.12.1 + CUDA 13.2 + Python 3.12 | documented | host driver and extension compile compatibility |

Once builds are run, the docs should record observed results and exact failure
points if any lane fails.

## Known Risks

- CUDA 13.x may expose stricter headers or compiler behavior than the current
  project has been tested against.
- The rasterizer header uses fixed-width integer types in CUDA headers; upstream
  CUDA 12 discussions mention explicit header includes may be needed in some
  configurations.
- `simple-knn` uses `FLT_MAX`; this checkout already includes `<cfloat>`, which
  addresses a common newer-CUDA build issue.
- PyTorch 2.12 extension ABI compatibility must be proven by building the image.
- `full_eval.py` shells out through string concatenation and writes `timing.txt`
  even when training variables are not initialized for some skip modes. This is
  a readiness finding, but fixing it is outside this initial Docker asset scope
  unless it blocks documented benchmark commands.
- Metrics use LPIPS and torchvision pretrained weights. Reproducible offline
  runs require a persistent or pre-populated cache.
- The project seeds Python, NumPy, and torch, but full GPU determinism is not
  guaranteed by the existing optimizer.

## Acceptance Criteria

The implementation is complete when:

- a detailed project/Docker training and benchmarking document exists;
- `Dockerfile` builds from the requested PyTorch image family;
- dependency constraints do not override torch packages supplied by the base
  image;
- CUDA submodules are installed during build as non-editable packages;
- `docker-compose.yml` exposes datasets, runs, artifacts, and cache mounts;
- runtime helper scripts exist and are copied into the image;
- the no-dataset smoke path is documented and runnable inside the container;
- any attempted Docker build or runtime verification is reported with exact
  command outcome.

## Implementation Sequence

1. Add requirements files.
2. Add Docker helper scripts.
3. Add Dockerfile.
4. Add Docker Compose file.
5. Add Docker training and benchmarking documentation.
6. Run static checks.
7. Attempt Docker build and smoke verification if Docker is available and the
   build cost is acceptable.
8. Update documentation with verified state or observed blockers.
