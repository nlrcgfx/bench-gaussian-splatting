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
