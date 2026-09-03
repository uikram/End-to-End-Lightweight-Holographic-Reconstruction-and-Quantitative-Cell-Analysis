"""Computational-efficiency profiling for the two imaging configurations.

Reports the quantities that decide whether the framework can run outside a
workstation: parameter count, multiply-accumulate cost, median and tail latency,
throughput and peak device memory, under both native PyTorch and a compiled ONNX
Runtime graph.

Latency is measured on synthetic input of the configured field size. Timing does
not depend on image content, so this isolates the architecture and runtime from
dataset loading, while the accuracy differences between modalities come from the
evaluation path instead.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..models import ExportWrapper
from ..utils import count_parameters, get_logger, run_name

LOGGER = get_logger(__name__)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_flops(model: torch.nn.Module, input_shape: tuple, device: torch.device) -> dict:
    """Multiply-accumulate count via thop, if it is installed."""
    try:
        from thop import profile
    except ImportError:
        LOGGER.info("thop is not installed; skipping the MAC count")
        return {}

    dummy = torch.randn(*input_shape, device=device)
    wrapper = ExportWrapper(model).to(device).eval()
    try:
        macs, _ = profile(wrapper, inputs=(dummy,), verbose=False)
        return {"gmacs": float(macs) / 1e9}
    except Exception as exc:                      # thop lacks rules for some ops
        LOGGER.warning("MAC counting failed: %s", exc)
        return {}


def benchmark_pytorch(
    model: torch.nn.Module, cfg: Config, device: torch.device, precision: str
) -> dict:
    benchmark_cfg = cfg.deploy.benchmark
    size = benchmark_cfg.input_size
    shape = (benchmark_cfg.batch_size, cfg.model.in_channels, size, size)

    model = model.to(device).eval()
    dummy = torch.randn(*shape, device=device)

    if precision == "fp16":
        if device.type != "cuda":
            LOGGER.info("fp16 timing needs CUDA; skipping")
            return {}
        model = model.half()
        dummy = dummy.half()

    with torch.inference_mode():
        for _ in range(benchmark_cfg.warmup_runs):
            model(dummy)
        _synchronise(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        timings = []
        for _ in range(benchmark_cfg.timed_runs):
            _synchronise(device)
            started = time.perf_counter()
            model(dummy)
            _synchronise(device)
            timings.append((time.perf_counter() - started) * 1000.0)

    result = _summarise(timings, benchmark_cfg.batch_size)
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)

    if precision == "fp16":
        model.float()
    return result


def _preload_cuda_libraries() -> int:
    """Load the CUDA and cuDNN shared objects that pip installs into site-packages.

    onnxruntime resolves its CUDA provider through the system dynamic loader,
    which searches the usual library paths but not site-packages. PyTorch ships
    its CUDA runtime as pip wheels under ``nvidia/*/lib`` and loads them itself,
    so torch runs on the GPU while onnxruntime reports
    "libcudnn_adv.so.9: cannot open shared object file" and falls back to CPU on
    the very same machine.

    Loading the libraries into this process with RTLD_GLOBAL first makes them
    resolvable by soname when the provider is created. Two passes, because the
    cuDNN sub-libraries depend on one another and a first-pass failure often
    succeeds once its dependency is resident.
    """
    import ctypes
    import glob
    import os

    try:
        import nvidia
    except ImportError:
        return 0

    # `nvidia` is a namespace package: pip installs nvidia-cudnn-cu12 and friends
    # as separate distributions sharing the name, so it has no __init__.py and
    # its __file__ is None. The directories live on __path__ instead.
    roots = [str(entry) for entry in getattr(nvidia, "__path__", []) or []]
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    if not roots:
        return 0

    # cuDNN first: it is the one the CUDA provider names explicitly.
    order = ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cufft",
             "curand", "cusolver", "cusparse", "nvjitlink")

    candidates: list[str] = []
    for root in roots:
        for package in order:
            candidates.extend(sorted(glob.glob(os.path.join(root, package, "lib", "*.so*"))))

    loaded: set[str] = set()
    for _ in range(2):
        for path in candidates:
            if path in loaded:
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.add(path)
            except OSError:
                continue      # unresolved dependency; the second pass may fix it

    if loaded:
        LOGGER.info("preloaded %d CUDA/cuDNN libraries from %s", len(loaded), roots[0])
    return len(loaded)


def benchmark_onnx(onnx_path: str | Path, cfg: Config, device: torch.device) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        LOGGER.warning("onnxruntime is not installed; skipping the ONNX benchmark")
        return {}

    if device.type == "cuda":
        _preload_cuda_libraries()

    benchmark_cfg = cfg.deploy.benchmark
    size = benchmark_cfg.input_size

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    on_gpu = device.type == "cuda" and "CUDAExecutionProvider" in available
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if on_gpu
        else ["CPUExecutionProvider"]
    )

    if device.type == "cuda" and not on_gpu:
        # Silently falling back would put a CPU latency next to a GPU latency in
        # the same table, under the same column, differing by an order of
        # magnitude for a reason that has nothing to do with either model.
        LOGGER.warning(
            "onnxruntime has no CUDA provider (available: %s), so this ONNX timing "
            "runs on CPU while the PyTorch timing ran on %s. The two rows are NOT "
            "comparable. Install onnxruntime-gpu to time ONNX on the GPU, or drop "
            "'onnx' from deploy.benchmark.runtimes and report PyTorch only.",
            available, torch.cuda.get_device_name(0),
        )

    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)

    # Being listed in get_available_providers() only means the wheel was built
    # with CUDA; the provider can still fail to instantiate when its shared
    # libraries do not load, and onnxruntime then falls back to CPU without
    # raising. Only the session's own provider list says what actually ran.
    if on_gpu and "CUDAExecutionProvider" not in session.get_providers():
        LOGGER.warning(
            "CUDAExecutionProvider was requested and advertised, but the session "
            "fell back to %s. The ONNX timing is therefore on CPU and is NOT "
            "comparable to the PyTorch row. The usual cause is a CUDA/cuDNN "
            "version mismatch: check the onnxruntime error above for the exact "
            "missing library.",
            session.get_providers(),
        )

    input_meta = session.get_inputs()[0]
    dtype = np.float16 if "float16" in input_meta.type else np.float32
    dummy = np.random.randn(benchmark_cfg.batch_size, cfg.model.in_channels, size, size).astype(dtype)
    feed = {input_meta.name: dummy}

    for _ in range(benchmark_cfg.warmup_runs):
        session.run(None, feed)

    timings = []
    for _ in range(benchmark_cfg.timed_runs):
        started = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - started) * 1000.0)

    result = _summarise(timings, benchmark_cfg.batch_size)
    actual = session.get_providers()
    result["providers"] = ",".join(actual)
    # Explicit flag so a reader of the CSV cannot miss a cross-device comparison.
    result["comparable_to_pytorch_row"] = bool(
        device.type != "cuda" or "CUDAExecutionProvider" in actual
    )
    return result


def _summarise(timings: list[float], batch_size: int) -> dict:
    array = np.asarray(timings)
    mean = float(array.mean())
    return {
        "latency_mean_ms": mean,
        "latency_p50_ms": float(np.percentile(array, 50)),
        "latency_p99_ms": float(np.percentile(array, 99)),
        "latency_std_ms": float(array.std()),
        "fps": float(1000.0 * batch_size / mean) if mean > 0 else float("nan"),
        "timed_runs": int(array.size),
    }


def profile_model(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    modality: str,
    onnx_dir: str | Path | None = None,
) -> list[dict]:
    """Run every configured runtime/precision combination for one model."""
    from .export import export_onnx

    benchmark_cfg = cfg.deploy.benchmark
    size = benchmark_cfg.input_size
    shape = (benchmark_cfg.batch_size, cfg.model.in_channels, size, size)

    base = {
        "experiment": cfg.experiment_name,
        "modality": modality,
        "encoder": cfg.model.encoder,
        "frontend": cfg.model.frontend.kind,
        "lora": cfg.model.lora.enabled,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "input_size": size,
        "torch_version": torch.__version__,
    }
    base.update({f"params_{k}": v for k, v in count_parameters(model).items()})
    if benchmark_cfg.measure_flops:
        base.update(measure_flops(model, shape, device))

    rows: list[dict] = []
    onnx_dir = Path(onnx_dir) if onnx_dir else None

    for precision in benchmark_cfg.precisions:
        for runtime in benchmark_cfg.runtimes:
            LOGGER.info("benchmarking %s | %s | %s", modality, runtime, precision)

            if runtime == "pytorch":
                stats = benchmark_pytorch(model, cfg, device, precision)
            elif runtime == "onnx":
                if onnx_dir is None:
                    LOGGER.warning("no onnx_dir given; skipping the ONNX runtime")
                    continue
                path = onnx_dir / f"{run_name(cfg.experiment_name, modality)}_{precision}.onnx"
                try:
                    export_onnx(model, cfg, path, size, device=device, precision=precision)
                    stats = benchmark_onnx(path, cfg, device)
                except Exception as exc:
                    LOGGER.warning("ONNX path failed for %s/%s: %s", modality, precision, exc)
                    stats = {}
            else:
                raise ValueError(f"unknown runtime {runtime!r}")

            if not stats:
                continue

            row = dict(base)
            row.update({"runtime": runtime, "precision": precision})
            row.update(stats)
            rows.append(row)

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return rows
