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
    for attr_name in ("GaussianRasterizationSettings", "GaussianRasterizer"):
        getattr(module, attr_name)
    print("diff_gaussian_rasterization imports ok")
    if hasattr(module, "SparseGaussianAdam"):
        print("diff_gaussian_rasterization SparseGaussianAdam optional import ok")
    else:
        print("diff_gaussian_rasterization SparseGaussianAdam optional import unavailable")


def main() -> int:
    require_cuda()
    check_rasterizer_imports()
    check_simple_knn()
    check_fused_ssim()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
