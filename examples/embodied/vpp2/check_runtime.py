#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check that training uses the image runtime and can initialize CUDA."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def inspect_runtime(require_cuda=False):
    import torch

    packages = {}
    for name in ("torch", "torchvision", "deepspeed", "accelerate", "numpy", "av", "transformers", "diffusers"):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version, "location": str(dist.locate_file(""))}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_path": torch.__file__,
        "packages": packages,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PYTHONPATH",
                "VIRTUAL_ENV",
                "CONDA_PREFIX",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
                "FLASH_ATTENTION_DETERMINISTIC",
                "NCCL_ALGO",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "PYTHONHASHSEED",
            )
        },
        "cuda_checked": require_cuda,
    }
    errors = []
    if sys.prefix != sys.base_prefix:
        errors.append("LoongForge must use the image system Python, not a community virtualenv")

    def from_virtualenv(value):
        path = Path(value).resolve()
        return any((parent / "pyvenv.cfg").is_file() for parent in (path, *path.parents))

    if any(from_virtualenv(item) for item in sys.path if item):
        errors.append("Community virtualenv/site-packages leaked into target sys.path")
    if require_cuda:
        try:
            torch.cuda.init()
            report["cuda_device_count"] = torch.cuda.device_count()
        except Exception as exc:
            errors.append(f"System CUDA initialization failed: {exc}")
    report["errors"] = errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_runtime(args.require_cuda)
    rendered = json.dumps(report, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
