# Docker Training And Benchmarking

This document describes the project structure and the Docker workflow for
reproducible 3D Gaussian Splatting training, rendering, and benchmarking.

The Docker lane is intentionally focused on the Python optimizer, renderer,
metrics scripts, and CUDA extension stack.

## Project Overview

This repository follows the GraphDECO 3D Gaussian Splatting layout. The main
Python workflow starts from a COLMAP or NeRF synthetic dataset, optimizes a
Gaussian scene representation, renders train/test image sets, then computes
image quality metrics.

- `train.py` optimizes 3D Gaussians from COLMAP or NeRF synthetic inputs.
- `render.py` renders train/test image sets from a saved model.
- `metrics.py` computes SSIM, PSNR, and LPIPS from rendered outputs.
- `full_eval.py` orchestrates the paper-style train, render, and metrics workflow.
- `convert.py` prepares COLMAP datasets from input images when COLMAP and ImageMagick are available.
- `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, and `submodules/fused-ssim` are CUDA extensions compiled during Docker image build.

The checked-in `env.yml` remains useful as historical project context, but the
Docker image does not use Conda. The image inherits Python, torch, torchvision,
CUDA runtime wheels, and CUDA development tooling from an official PyTorch
container, then builds the repository's CUDA extension submodules against that
PyTorch environment.

## Runtime Components

The Python runtime has five practical layers:

| Layer | Files | Role |
| --- | --- | --- |
| Training entry point | `train.py` | Loads a scene, optimizes Gaussians, writes checkpoints, point clouds, config, and TensorBoard events. |
| Renderer entry point | `render.py` | Loads a trained model and renders train/test sets under the model path. |
| Metrics entry point | `metrics.py` | Reads rendered images and ground truth, then writes `results.json` and `per_view.json`. |
| Benchmark orchestration | `full_eval.py` | Runs training, rendering, and metrics across the standard MipNeRF360, Tanks and Temples, and Deep Blending scene lists. |
| CUDA extensions | `submodules/*` | Provide differentiable rasterization, nearest-neighbor distance computation, fused SSIM, and optional sparse Adam support. |

The CUDA extension packages are installed into the image as non-editable
packages with `pip install --no-build-isolation`. This makes the image a
reproducible training artifact: the extension binaries are built once during
image construction rather than at container startup.

The Dockerfile lives at `docker/Dockerfile` and applies one checked-in
compatibility patch after copying the source into the image:
`docker/patches/diff-gaussian-rasterization-cstdint.patch`. The patch inserts
`#include <cstdint>` into the copied `diff-gaussian-rasterization` rasterizer
header. This is required by the PyTorch 2.12 / CUDA 13 compiler stack because
the extension uses fixed-width integer types that are not pulled in
transitively. The patch is applied with `git apply --ignore-space-change`, so
the CRLF line endings used by the current submodule checkout are tolerated, but
image builds still fail if the patch no longer applies cleanly. The checked-out
submodule source is not modified by this Docker build step.

## Dataset And Output Layout

The container uses stable paths so host automation can mount datasets, collect
outputs, and persist caches across image rebuilds.

| Container Path | Purpose |
| --- | --- |
| `/workspace/datasets` | Mounted COLMAP, NeRF synthetic, or benchmark datasets |
| `/workspace/runs` | Training model outputs, checkpoints, TensorBoard event files |
| `/workspace/artifacts` | Rendered benchmark artifacts, copied reports, exported summaries |
| `/workspace/cache` | Torch, torchvision, LPIPS, pip, and model weight cache |
| `/opt/gaussian-splatting` | Copied project source and installed CUDA extension sources |
| `/workspace/project` | Default command working directory |

The project scripts already support explicit paths. Prefer passing both
`-s/--source_path` and `-m/--model_path` in Docker commands:

- Use `-s /workspace/datasets/<scene>` for mounted input datasets.
- Use `-m /workspace/runs/<run-name>` for training outputs.

For COLMAP datasets, the expected structure is the upstream layout:

```text
<scene>/
  images/
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

Text COLMAP files are also supported by the project loaders. Optional depth
regularization expects depth maps and `sparse/0/depth_params.json`, as described
in the upstream README.

## Docker Image Lanes

Docker Buildx Bake defines the reproducible build matrix in `docker-bake.hcl`.
The default lane is:

```text
cu130 -> pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel
```

Additional runtime-verified lanes are:

```text
cu126 -> pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel
cu132 -> pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel
```

Bake tags the lane images as `gaussian-splatting:pytorch-2.12.1-cu126`,
`gaussian-splatting:pytorch-2.12.1-cu130`, and
`gaussian-splatting:pytorch-2.12.1-cu132`.

The CUDA 13.0 and CUDA 13.2 lanes default to:

```text
8.6;8.9;12.0
```

This covers Ampere, Ada, and Blackwell-oriented builds. Override
`TORCH_CUDA_ARCH_LIST` if your deployment target needs a different compiled
architecture set.

The CUDA 12.6 lane defaults to:

```text
8.6;8.9
```

CUDA 12.6's `nvcc` does not support `sm_120`, so the `cu126` Bake target uses
`TORCH_CUDA_ARCH_LIST_CU126` for its lane-specific default. Override that
variable if you need a different CUDA 12.6 architecture set.

## Build Commands

Builds should use Docker Buildx Bake. Compose remains the preferred runtime
interface for mounted datasets, logs, artifacts, shared memory, and GPUs.

Inspect the resolved Bake graph without building:

```powershell
docker buildx bake --print
```

Build and load the verified CUDA 13.0 lane locally:

```powershell
docker buildx bake cu130 --load
```

Build and load the CUDA 12.6 lane locally:

```powershell
docker buildx bake cu126 --load
```

Build and load the CUDA 13.2 lane locally:

```powershell
docker buildx bake cu132 --load
```

Build all lanes without loading them into the local Docker image store:

```powershell
docker buildx bake all
```

Build all lanes and load them into the local Docker image store:

```powershell
docker buildx bake all --load
```

Tune parallel extension compilation with `MAX_JOBS`:

```powershell
docker buildx bake cu130 --load --set cu130.args.MAX_JOBS=4
```

Bake exports inline cache metadata by default. This keeps local `--load`
workflows stable on Docker Desktop while preserving reusable BuildKit metadata
in the built image. The Dockerfile also uses BuildKit cache mounts for pip and
is layered so CUDA extension submodule sources and `docker/patches/` are copied
before the ordinary Python runtime files. This keeps extension wheel builds
cacheable when only training scripts, helpers, or documentation change.

`.buildx-cache/` is ignored so local-cache experiments can be done with
`--set` overrides without polluting the repository, but local cache directories
are not the default because exporting into the same local cache path can trigger
Docker Desktop ref-lock errors.

## Runtime Checks

Run the runtime information report:

```powershell
docker compose run --rm gaussian-splatting pytorch-cuda-runtime-info
```

Run the no-dataset smoke test:

```powershell
docker compose run --rm gaussian-splatting pytorch-cuda-smoke-test
```

Passing `pytorch-cuda-smoke-test` moves a lane from `build-verified` to
`runtime-verified`. The smoke test verifies:

- torch can see CUDA;
- a tiny CUDA tensor computation works;
- `diff_gaussian_rasterization` imports the required rasterizer classes;
- optional `SparseGaussianAdam` availability is reported but does not fail the default smoke test;
- `simple_knn._C.distCUDA2` runs on a small point tensor;
- `fused_ssim.fused_ssim` runs forward and backward on small image tensors.

## Training Workflows

Train a mounted COLMAP dataset:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden --disable_viewer
```

Train with an evaluation split:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-eval --eval --disable_viewer
```

Train with sparse Adam only when `pytorch-cuda-runtime-info` reports
`import_diff_gaussian_rasterization.SparseGaussianAdam: ok (optional)`.
The verified default CUDA 13.0 lane for this checkout currently reports that
optional import as unavailable, so default Adam is the reproducible path unless
the rasterizer submodule is replaced with one that exports `SparseGaussianAdam`.

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-fast --eval --optimizer_type sparse_adam --disable_viewer
```

Train with depth regularization:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-depth --eval -d depths2/ --disable_viewer
```

Train with exposure compensation:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-exp --eval --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp --disable_viewer
```

Train with antialiasing:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden-aa --eval --antialiasing --disable_viewer
```

## Rendering And Metrics

Render test views from a trained model:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/render.py -s /workspace/datasets/garden -m /workspace/runs/garden-eval --skip_train
```

Compute metrics:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/metrics.py -m /workspace/runs/garden-eval
```

Metrics may download torchvision and LPIPS weights into `/workspace/cache` on
first run. Keep the cache mount persistent for reproducible repeated benchmark
runs. For offline runs, pre-populate the cache before disabling network access.

## Full Benchmark Workflow

The repository's `full_eval.py` script expects the official benchmark dataset
groups to be mounted. Example:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/full_eval.py -m360 /workspace/datasets/mipnerf360 -tat /workspace/datasets/tandt -db /workspace/datasets/deep_blending --output_path /workspace/runs/full-eval
```

Use pretrained models and skip training:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/full_eval.py -m360 /workspace/datasets/mipnerf360 -tat /workspace/datasets/tandt -db /workspace/datasets/deep_blending --output_path /workspace/runs/pretrained --skip_training
```

Use already-rendered evaluation images and skip training plus rendering:

```powershell
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/full_eval.py --output_path /workspace/runs/eval-images --skip_training --skip_rendering
```

Read the readiness notes below before relying on `full_eval.py` for automated
CI-style benchmarking. The script is faithful to upstream behavior, but it is
less robust than invoking `train.py`, `render.py`, and `metrics.py` directly.
Its `--fast` option requests `--optimizer_type sparse_adam`, so use it only on
images where the optional sparse Adam import is available.

## Compose Reference

`docker-compose.yml` defines one service:

```text
gaussian-splatting
```

Useful host-side environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTORCH_IMAGE` | `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel` | PyTorch base image lane |
| `TORCH_CUDA_ARCH_LIST` | `8.6;8.9;12.0` | CUDA architectures compiled into extension binaries |
| `TORCH_CUDA_ARCH_LIST_CU126` | `8.6;8.9` | CUDA 12.6 Bake-lane architecture list |
| `MAX_JOBS` | `8` | Parallel jobs for extension builds |
| `GS_IMAGE` | `gaussian-splatting:pytorch-2.12.1-cu130` | Local image tag |
| `GS_DATASETS_HOST` | `./data` | Host dataset mount |
| `GS_RUNS_HOST` | `./runs` | Host training output mount |
| `GS_ARTIFACTS_HOST` | `./artifacts` | Host artifact mount |
| `GS_CACHE_HOST` | `./cache` | Host cache mount |
| `GS_SHM_SIZE` | `16gb` | Shared memory size for the service |

Start an interactive shell:

```powershell
docker compose run --rm gaussian-splatting bash
```

## Readiness Matrix

The readiness states are:

- `documented`: the lane is described and the image tag exists.
- `build-verified`: the image builds and all CUDA extensions compile.
- `runtime-verified`: runtime info and smoke test pass with a visible NVIDIA GPU.
- `blocked`: a verification command failed; see verification notes.

| Lane | Current State | Verification Required |
| --- | --- | --- |
| `cu126`: PyTorch 2.12.1 + CUDA 12.6 + Python 3.12 | runtime-verified | Re-run on target hardware and benchmark datasets before publishing results |
| `cu130`: PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 | runtime-verified | Re-run on target hardware and benchmark datasets before publishing results |
| `cu132`: PyTorch 2.12.1 + CUDA 13.2 + Python 3.12 | runtime-verified | Re-run on target hardware and benchmark datasets before publishing results |

## Known Risks And Readiness Notes

- CUDA 12.6, CUDA 13.0, and CUDA 13.2 are build-verified and runtime-verified in this workspace.
- CUDA 12.6 uses `TORCH_CUDA_ARCH_LIST_CU126=8.6;8.9` because CUDA 12.6's `nvcc` rejects `sm_120`.
- Sparse Adam is optional. The verified images for this checkout report `SparseGaussianAdam` as unavailable, so `--optimizer_type sparse_adam` and `full_eval.py --fast` are not part of the verified workflow.
- The checked-in rasterizer `<cstdint>` patch is required for CUDA 13 extension compilation with this rasterizer checkout.
- `full_eval.py` uses string-based `os.system` calls and is less robust than direct command invocation.
- `full_eval.py` can write `timing.txt` from uninitialized timing variables in some skip-only modes.
- Full GPU determinism is not guaranteed by the existing optimizer.
- Offline metrics runs require a populated `/workspace/cache`.
- The CUDA extension ABI must be proven per PyTorch image lane by building the image.
- Host drivers must support the selected CUDA runtime. CUDA 13.2 requires a sufficiently new NVIDIA driver.

## Verification Notes

The image lanes were verified on this workspace with:

```powershell
docker buildx bake cu126 --load
docker buildx bake cu130 --load
docker buildx bake cu132 --load
```

Then each loaded image was checked through Compose with `GS_IMAGE` set to the
lane tag and `--pull never`:

```powershell
$env:GS_IMAGE = 'gaussian-splatting:pytorch-2.12.1-cu126'
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-runtime-info
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-smoke-test

$env:GS_IMAGE = 'gaussian-splatting:pytorch-2.12.1-cu130'
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-runtime-info
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-smoke-test

$env:GS_IMAGE = 'gaussian-splatting:pytorch-2.12.1-cu132'
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-runtime-info
docker compose run --rm --pull never gaussian-splatting pytorch-cuda-smoke-test
```

Observed runtime summary:

```text
cu126: python 3.12.3, torch 2.12.1+cu126, torch_cuda 12.6, cudnn 91002, arch 8.6;8.9
cu130: python 3.12.3, torch 2.12.1+cu130, torch_cuda 13.0, cudnn 92000, arch 8.6;8.9;12.0
cu132: python 3.12.3, torch 2.12.1+cu132, torch_cuda 13.2, cudnn 92000, arch 8.6;8.9;12.0
all lanes: cuda_available True, NVIDIA GeForce RTX 4070 SUPER capability 8.9
all lanes: required imports ok for GaussianRasterizer, simple_knn.distCUDA2, fused_ssim
all lanes: SparseGaussianAdam unavailable as an optional import
all lanes: smoke test passed
```
