"""ONNX export for edge deployment.

LoRA residuals are folded into their base weights before tracing, so the exported
graph carries no adaptation branches and costs exactly what the frozen backbone
costs at inference.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from ..config import Config
from ..models import ExportWrapper
from ..utils import get_logger

LOGGER = get_logger(__name__)


def export_onnx(
    model: torch.nn.Module,
    cfg: Config,
    destination: str | Path,
    input_size: int,
    device: torch.device | None = None,
    precision: str = "fp32",
) -> Path:
    """Trace the model to ONNX and return the written path."""
    onnx_cfg = cfg.deploy.onnx
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cpu")
    model = model.to(device).eval()

    if onnx_cfg.merge_lora_before_export and getattr(model, "has_lora", False):
        model.merge_lora()

    wrapper = ExportWrapper(model).to(device).eval()

    in_channels = cfg.model.in_channels
    dummy = torch.randn(
        cfg.deploy.benchmark.batch_size, in_channels, input_size, input_size, device=device
    )

    if precision == "fp16":
        wrapper = wrapper.half()
        dummy = dummy.half()

    dynamic_axes = None
    if onnx_cfg.dynamic_axes:
        dynamic_axes = {onnx_cfg.input_name: {0: "batch", 2: "height", 3: "width"}}
        for name in onnx_cfg.output_names:
            dynamic_axes[name] = {0: "batch"}

    export_kwargs = dict(
        export_params=True,
        opset_version=onnx_cfg.opset,
        do_constant_folding=onnx_cfg.constant_folding,
        input_names=[onnx_cfg.input_name],
        output_names=list(onnx_cfg.output_names),
        dynamic_axes=dynamic_axes,
    )

    # Torch 2.5+ defaults to the dynamo exporter, which needs the optional
    # `onnxscript` package. The TorchScript path produces the same graph here and
    # has no extra dependency, so ask for it explicitly where the option exists.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(wrapper, dummy, str(destination), **export_kwargs)

    LOGGER.info(
        "exported ONNX (%s, opset %d, %dx%d) -> %s",
        precision, onnx_cfg.opset, input_size, input_size, destination,
    )
    return destination


def verify_onnx(path: str | Path) -> bool:
    """Structural check of an exported graph; False if onnx is unavailable."""
    try:
        import onnx
    except ImportError:
        LOGGER.warning("onnx is not installed; skipping graph verification")
        return False

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    LOGGER.info("ONNX graph verified: %d nodes", len(graph.graph.node))
    return True
