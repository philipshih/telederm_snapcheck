"""Export the trained SnapCheck quality model to deployment formats."""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import torch

from snapcheck.paths import MODEL_DIR
from snapcheck.quality_model import load_quality_checkpoint

LOGGER = logging.getLogger("snapcheck.export")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _ensure_cpu(model: torch.nn.Module) -> torch.nn.Module:
    return model.to(torch.device("cpu"))


def _export_torchscript(model: torch.nn.Module, output_dir: Path, quantization: str | None = None) -> List[Path]:
    outputs: List[Path] = []
    cpu_model = _ensure_cpu(model).eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(cpu_model, example)
    ts_path = output_dir / "snapcheck_quality.ts"
    traced.save(ts_path)
    outputs.append(ts_path)
    LOGGER.info("Saved TorchScript module to %s", ts_path)
    if quantization == "dynamic":
        quant_model = torch.quantization.quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)
        traced_q = torch.jit.trace(quant_model, example)
        ts_q_path = output_dir / "snapcheck_quality_dynamic_q.ts"
        traced_q.save(ts_q_path)
        outputs.append(ts_q_path)
        LOGGER.info("Saved dynamically quantized TorchScript module to %s", ts_q_path)
    return outputs


def _export_onnx(model: torch.nn.Module, output_dir: Path, opset: int = 17) -> Path:
    cpu_model = _ensure_cpu(model).eval()
    example = torch.randn(1, 3, 224, 224)
    onnx_path = output_dir / "snapcheck_quality.onnx"
    torch.onnx.export(
        cpu_model,
        example,
        onnx_path,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    )
    LOGGER.info("Saved ONNX model to %s", onnx_path)
    return onnx_path


def _export_tflite(onnx_path: Path, output_dir: Path, quantization: str | None = None) -> Path | None:
    try:
        import onnx  # type: ignore
        from onnx_tf.backend import prepare  # type: ignore
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        LOGGER.warning("Skipping TFLite export: %s", exc)
        return None

    onnx_model = onnx.load(onnx_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tmp_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        if quantization == "dynamic":  # pragma: no cover - depends on TF build
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    tflite_path = output_dir / "snapcheck_quality.tflite"
    tflite_path.write_bytes(tflite_model)
    LOGGER.info("Saved TFLite model to %s", tflite_path)
    return tflite_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SnapCheck quality model")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "snapcheck_quality.pt")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "exports")
    parser.add_argument("--formats", nargs="*", default=None, help="Subset of formats to export")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load checkpoint")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    model, config, label_columns = load_quality_checkpoint(args.checkpoint, device=args.device)
    target_formats: List[str] = args.formats or config.get("target_formats", ["torchscript", "onnx"])
    quantization = config.get("quantization")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[str, str] = {}
    if "torchscript" in target_formats:
        paths = _export_torchscript(model, output_dir, quantization=quantization)
        exported["torchscript"] = ", ".join(str(p.name) for p in paths)
    onnx_path: Path | None = None
    if "onnx" in target_formats or "tflite" in target_formats:
        onnx_path = _export_onnx(model, output_dir)
        exported["onnx"] = onnx_path.name
    if "tflite" in target_formats and onnx_path is not None:
        tflite_path = _export_tflite(onnx_path, output_dir, quantization=quantization)
        if tflite_path:
            exported["tflite"] = tflite_path.name

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "label_columns": label_columns,
        "exported_formats": exported,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Wrote export manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
