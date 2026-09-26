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
    # The named device, not the current one: every other CUDA call in this
    # module is now explicit about which GPU it means.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# A timing run whose 99th percentile exceeds its median by more than this is
# reported as unstable. On an idle GPU this architecture measures p99/p50 = 1.01
# with a standard deviation of 0.02 ms; the value below is far above anything a
# quiet device produces and far below the 2.7-3.0 a contended one does.
#
# It is calibrated for the GPU, which is where the reported numbers are
# measured. A CPU run on a shared machine is inherently jittier and can trip it
# at a short `timed_runs`; that is the conservative direction -- it asks for a
# second look rather than passing a contaminated number -- but a CPU flag is not
# by itself evidence of contention.
_STABILITY_LIMIT = 1.25


# What a single idle process legitimately holds on a CUDA device beyond its
# caching allocator: the CUDA context, the cuDNN and cuBLAS handles and their
# workspaces. Measured on this machine at 400-600 MiB for this model.
#
# It exists because the memory-arithmetic fallback below cannot see whose memory
# it is. The FIRST version of this function could not either, and charged this
# process's own context to "other processes": every one of the twenty efficiency
# rows in the 2026-09-17 run came back with `device_idle_at_start = False` on a
# GPU that was demonstrably idle -- `timing_stable` was true and p99/p50 ran
# 1.00-1.11 in the same rows -- and the results table refused to quote any of
# them. A false alarm that suppresses good measurements is not the safe
# direction; it is the same failure as a missed alarm, pointing the other way.
_SELF_CONTEXT_ALLOWANCE_MB = 1024.0


def foreign_memory_mb(device: torch.device) -> float:
    """Device memory held by processes other than this one, in MiB.

    Nothing in this module could originally tell an idle GPU from a busy one, and
    that is not a hypothetical concern: a benchmark of this model was once run
    while a training job occupied the same device, which inflated the tail
    latencies enough to make the COMPACT decoder look slower than the full one
    and so inverted the efficiency conclusion. The numbers looked entirely
    ordinary.

    Per-process attribution comes from ``nvidia-smi --query-compute-apps``, which
    lists every process on the device with its PID, so this process can be
    excluded by identity rather than by an allowance. That is the only way to
    answer the question the caller is actually asking. The memory-arithmetic
    fallback below is used when nvidia-smi is absent or its output cannot be
    attributed to this device, and it subtracts a fixed allowance for this
    process's own context because it has no PID to filter on.
    """
    if device.type != "cuda":
        return 0.0

    attributed = _foreign_memory_from_nvidia_smi(device)
    if attributed is not None:
        return attributed

    try:
        # THE REQUESTED DEVICE, not the current one. Both calls default to the
        # current CUDA device, and resolve_device never calls set_device -- so
        # `--device cuda:1` measured GPU 0's occupancy and could report a busy
        # GPU 1 as idle, which is exactly the contention this exists to catch.
        free, total = torch.cuda.mem_get_info(device)
        mine = torch.cuda.memory_reserved(device)
    except Exception:
        return 0.0
    used_by_others = (total - free - mine) / (1024 ** 2)
    return max(0.0, used_by_others - _SELF_CONTEXT_ALLOWANCE_MB)


def _foreign_memory_from_nvidia_smi(device: torch.device) -> float | None:
    """MiB held on ``device`` by PIDs other than this one, or None if unknown.

    Returns None rather than 0.0 whenever the answer cannot be established --
    nvidia-smi missing, a non-zero exit, unparseable output, or a device whose
    UUID torch does not expose -- so the caller falls back instead of treating an
    unanswered question as an idle device.
    """
    import os
    import subprocess

    try:
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    except Exception:
        return None
    if uuid is None:
        return None
    # torch returns a uuid.UUID; nvidia-smi prints "GPU-<hex-with-dashes>".
    wanted = f"GPU-{uuid}".lower()

    try:
        completed = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None

    own_pid = os.getpid()
    total_mb = 0.0
    matched_device = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        gpu_uuid, pid, used = fields
        if gpu_uuid.lower() != wanted:
            continue
        matched_device = True
        try:
            if int(pid) == own_pid:
                continue
            total_mb += float(used)
        except ValueError:
            # "[N/A]" appears under MIG and in some container configurations.
            # An unreadable row means the question is unanswered for this
            # device, so fall back rather than silently undercount.
            return None
    # No row for this device is a legitimate answer only if this process itself
    # appears nowhere either -- which it must, since it has a live context. If
    # the device never matched, the UUID comparison is wrong and the result
    # would be a confident "idle" for a device that was never examined.
    if not matched_device:
        return None
    return total_mb


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
    import copy

    benchmark_cfg = cfg.deploy.benchmark
    size = benchmark_cfg.input_size
    shape = (benchmark_cfg.batch_size, cfg.model.in_channels, size, size)

    model = model.to(device).eval()
    dummy = torch.randn(*shape, device=device)

    if precision == "fp16":
        if device.type != "cuda":
            LOGGER.info("fp16 timing needs CUDA; skipping")
            return {}
        # A COPY, because .half() mutates in place. This used to convert the
        # caller's model down and then back up with .float(), which round-trips
        # every weight and every BatchNorm statistic through 10 mantissa bits.
        # The model is reused afterwards -- profile_model exports it to ONNX at
        # fp16 next, and that ONNX file is a deliverable -- so the degradation
        # was being written to disk.
        model = copy.deepcopy(model).half()
        dummy = dummy.half()

    with torch.inference_mode():
        for _ in range(benchmark_cfg.warmup_runs):
            model(dummy)
        _synchronise(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            # The allocation already standing at this point, so the peak can be
            # reported as WEIGHTS + ACTIVATIONS rather than as whatever else
            # happens to be resident. It matters for the fp16 row: that runs on
            # a deep copy (the caller's model must not be converted in place),
            # so the fp32 original is still allocated and a raw
            # max_memory_allocated would charge this model roughly 3x its weight
            # bytes and make the fp16 row incomparable with the fp32 one.
            baseline_bytes = torch.cuda.memory_allocated(device)

        timings = []
        for _ in range(benchmark_cfg.timed_runs):
            _synchronise(device)
            started = time.perf_counter()
            model(dummy)
            _synchronise(device)
            timings.append((time.perf_counter() - started) * 1000.0)

    result = _summarise(timings, benchmark_cfg.batch_size, device.type)
    if device.type == "cuda":
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        weight_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
        activation_bytes = max(
            0, torch.cuda.max_memory_allocated(device) - baseline_bytes
        )
        result["weights_mb"] = weight_bytes / (1024 ** 2)
        result["peak_activation_mb"] = activation_bytes / (1024 ** 2)
        # What a deployment would need for THIS model: its own weights plus the
        # transient peak of one forward pass.
        result["peak_allocated_mb"] = (weight_bytes + activation_bytes) / (1024 ** 2)
        # Kept for continuity, but it describes the whole process and so
        # includes anything else this benchmark is holding.
        result["process_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        result["process_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
    # No `model.float()` restore is needed any more: fp16 runs on a deep copy,
    # so the caller's model was never converted.
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

    # The ONNX session's own device, which is what its timing describes -- it
    # can fall back to CPU while `device` says cuda.
    on_cuda = "CUDAExecutionProvider" in session.get_providers()
    result = _summarise(
        timings, benchmark_cfg.batch_size, "cuda" if on_cuda else "cpu"
    )
    actual = session.get_providers()
    result["providers"] = ",".join(actual)
    # Explicit flag so a reader of the CSV cannot miss a cross-device comparison.
    result["comparable_to_pytorch_row"] = bool(
        device.type != "cuda" or "CUDAExecutionProvider" in actual
    )
    return result


def _summarise(timings: list[float], batch_size: int,
               device_type: str = "cuda") -> dict:
    array = np.asarray(timings)
    mean = float(array.mean())
    p50 = float(np.percentile(array, 50))
    p99 = float(np.percentile(array, 99))
    ratio = p99 / p50 if p50 > 0 else float("nan")
    stable = bool(np.isfinite(ratio) and ratio <= _STABILITY_LIMIT)
    if not stable:
        # Said out loud, because a contaminated latency is indistinguishable from
        # a real one in the CSV. p50 survives contention roughly intact while the
        # tail does not, so the ratio is the signal.
        # The cause differs by device, and naming the wrong one sends someone
        # hunting for a job that was never there. On the GPU a tripped ratio
        # means contention; on the CPU it is usually ordinary scheduler jitter.
        LOGGER.warning(
            "latency distribution is not stable: p99/p50 = %.2f (limit %.2f), "
            "p50 %.2f ms, p99 %.2f ms, sd %.3f ms. %s",
            ratio, _STABILITY_LIMIT, p50, p99, float(array.std()),
            "Something else was almost certainly using this GPU: re-measure "
            "with nothing else running before reporting these numbers."
            if device_type == "cuda" else
            "On a shared CPU this is usually scheduler jitter rather than "
            "contention; raise deploy.benchmark.timed_runs if it persists.",
        )
    return {
        "latency_mean_ms": mean,
        "latency_p50_ms": p50,
        "latency_p99_ms": p99,
        "latency_std_ms": float(array.std()),
        # Carried into the CSV so collect_results can refuse to tabulate a
        # contaminated row instead of printing it as a measurement.
        "latency_p99_over_p50": ratio,
        "timing_stable": stable,
        "timing_device": device_type,
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

    # CHECKED ONCE, BEFORE ANY TIMING, AND RECORDED IN EVERY ROW.
    #
    # A latency measured while another job holds the same GPU is not a property
    # of the model, and it is not visible in the number. It happened to this
    # study: stage 8 overlapped a training run on the same device, the tail
    # latencies roughly tripled, and the compact decoder came out slower than the
    # full one -- the opposite of the truth. Both the warning and the recorded
    # value exist so that cannot pass unnoticed again.
    # Recorded so a reader can tell a PID-attributed zero from a fallback
    # estimate that merely came out below the allowance.
    attributed = _foreign_memory_from_nvidia_smi(device)
    foreign_mb = attributed if attributed is not None else foreign_memory_mb(device)
    if foreign_mb > 256.0:
        LOGGER.warning(
            "another process is holding about %.0f MiB on this GPU. Latency "
            "measured now describes a contended device, not this model. Stop the "
            "other job, or run this stage on an idle GPU, before reporting any "
            "efficiency number.",
            foreign_mb,
        )

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
        "foreign_gpu_memory_mb": round(foreign_mb, 1),
        "device_idle_at_start": bool(foreign_mb <= 256.0),
        # "pid" means nvidia-smi attributed the memory process by process and
        # this process was excluded by its own PID. "allocator_arithmetic" means
        # that was unavailable and the number is free/total memory minus a fixed
        # allowance for this process's context, which is an estimate. A caution
        # from the estimate is not, on its own, grounds to discard a timing.
        "device_occupancy_source": (
            "pid" if attributed is not None else "allocator_arithmetic"
        ),
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
