# 3D Gaussian Splatting

Developer-focused fork of GraphDECO's 3D Gaussian Splatting codebase for
training, rendering, metrics, and Docker-based benchmarking.

Upstream project links:

- [Project page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Paper](https://repo-sam.inria.fr/fungraph/3d_gaussian_splatting_high.pdf)
- [Tanks and Temples / Deep Blending COLMAP data](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
- [Pretrained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
- [Evaluation images](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)

## What Is Here

- `train.py`: optimize 3D Gaussians from COLMAP or NeRF synthetic inputs.
- `render.py`: render train/test views from a trained model.
- `metrics.py`: compute SSIM, PSNR, and LPIPS for rendered outputs.
- `full_eval.py`: run the paper-style train/render/metrics workflow.
- `convert.py`: create a COLMAP-style dataset from input images.
- `submodules/diff-gaussian-rasterization`: differentiable CUDA rasterizer.
- `submodules/simple-knn`: CUDA nearest-neighbor distance extension.
- `submodules/fused-ssim`: fused SSIM CUDA extension.
- `docker/`, `docker-bake.hcl`, `docker-compose.yml`: reproducible Docker training lanes.

## Clone

```bash
git clone --recursive https://github.com/ramsafin/3d-gaussian-splatting.git
cd 3d-gaussian-splatting
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Dataset Layout

Training expects a COLMAP or NeRF synthetic dataset. COLMAP inputs should look
like this:

```text
<scene>/
  images/
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

Text COLMAP files are also supported. Use `convert.py` when starting from raw
input images.

## Docker Quick Start

Docker is the preferred path for reproducible training and benchmarking. The
default verified lane is PyTorch 2.12.1 + CUDA 13.0 + Python 3.12.

Build and load the default image:

```bash
docker buildx bake cu130 --load
```

Check runtime readiness:

```bash
docker compose run --rm gaussian-splatting pytorch-cuda-runtime-info
docker compose run --rm gaussian-splatting pytorch-cuda-smoke-test
```

Train a mounted dataset:

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/train.py -s /workspace/datasets/garden -m /workspace/runs/garden --disable_viewer
```

Render and score:

```bash
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/render.py -s /workspace/datasets/garden -m /workspace/runs/garden --skip_train
docker compose run --rm gaussian-splatting python /opt/gaussian-splatting/metrics.py -m /workspace/runs/garden
```

Default host mounts:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `./data` | `/workspace/datasets` | Input datasets |
| `./runs` | `/workspace/runs` | Training outputs |
| `./artifacts` | `/workspace/artifacts` | Reports and rendered artifacts |
| `./cache` | `/workspace/cache` | Torch, LPIPS, pip, and model caches |

Useful overrides:

```bash
GS_DATASETS_HOST=D:/datasets GS_RUNS_HOST=D:/runs docker compose run --rm gaussian-splatting bash
```

See [docs/docker-training-benchmarking.md](docs/docker-training-benchmarking.md)
for CUDA 12.6 / 13.0 / 13.2 lanes, verification status, cache notes, and full
benchmark commands.

## Local Setup

Local setup is useful for development on the Python scripts and CUDA extensions.
Use a CUDA/PyTorch combination that matches your compiler toolchain. The
historical local path uses CUDA 11.8 and PyTorch 2.1.2.

Create the Conda environment:

```bash
conda env create --file env.yml
conda activate gaussian_splatting
```

Install PyTorch CUDA 11.8 wheels:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Build CUDA extensions:

```bash
pip install -U -e submodules/diff-gaussian-rasterization --no-build-isolation
pip install -U -e submodules/simple-knn --no-build-isolation
pip install -U -e submodules/fused-ssim --no-build-isolation
```

On Windows, use a Visual Studio C++ toolchain compatible with the selected CUDA
SDK before building the extensions.

## Core Commands

Train:

```bash
python train.py -s <dataset> -m <run-dir>
```

Train with evaluation split:

```bash
python train.py -s <dataset> -m <run-dir> --eval
```

Render:

```bash
python render.py -s <dataset> -m <run-dir> --skip_train
```

Metrics:

```bash
python metrics.py -m <run-dir>
```

Prepare a dataset from raw images:

```bash
python convert.py -s <dataset-root> --resize
```

Run the standard benchmark script:

```bash
python full_eval.py -m360 <mipnerf360> -tat <tanks-and-temples> -db <deep-blending> --output_path <runs-dir>
```

## Common Training Options

| Option | Use |
| --- | --- |
| `--eval` | Hold out test views for evaluation. |
| `--iterations <n>` | Override training iterations. Default is `30000`. |
| `--test_iterations -1` | Avoid periodic test-set evaluation during training. |
| `--resolution <value>` | Control input image scaling. |
| `--data_device cpu` | Reduce VRAM usage for large datasets. |
| `--optimizer_type sparse_adam` | Use sparse Adam when the rasterizer build exports it. |
| `--antialiasing` | Enable EWA antialiasing during training/rendering. |
| `-d <depth-dir>` | Enable depth regularization. |

## Optional Features

Training speed acceleration, depth regularization, exposure compensation, and
antialiasing are available in this checkout. See `results.md` for comparison
tables and use `python train.py --help` for the current argument list.

Depth regularization expects generated depth maps plus `depth_params.json`:

```bash
python utils/make_depth_scale.py --base_dir <colmap-dataset> --depths_dir <depth-map-dir>
python train.py -s <dataset> -m <run-dir> -d <depth-map-dir>
```

Exposure compensation:

```bash
python train.py -s <dataset> -m <run-dir> --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp
```

## Development Notes

- Prefer Docker for reproducible benchmark results.
- Keep datasets, runs, artifacts, and caches outside the image.
- Rebuild CUDA extensions after changing compiler, CUDA, PyTorch, or extension source.
- Use `python train.py --help`, `python render.py --help`, and `python convert.py --help` for exact current flags.
- Metrics may download LPIPS and torchvision weights on first run; persist `./cache` or `/workspace/cache`.

## Citation

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimkuehler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

See [LICENSE.md](LICENSE.md).
