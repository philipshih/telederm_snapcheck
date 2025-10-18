from __future__ import annotations

import json
import hashlib
import re
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .paths import QUALITY_DATA_DIR, REPORT_DIR
from .quality_model import QualityModel, TARGET_LABELS
from .utils import configure_logging
from .vlm import VLMResponse, load_vlm_backend

LOGGER = configure_logging("snapcheck.triage")


_CACHE_EXCLUDE_KEYS = {"api_key"}


def _sanitize_backend_label(cfg: Dict[str, Any]) -> str:
    label = cfg.get("name") or cfg.get("model") or cfg.get("type") or "backend"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label.lower())


def _prepare_cache_dir(root: Path, cfg: Dict[str, Any]) -> Path:
    path = root / _sanitize_backend_label(cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_file_path(root: Path, backend_cfg: Dict[str, Any], image_rel_path: str, prompt: str) -> Path:
    sanitized = {k: v for k, v in backend_cfg.items() if k not in _CACHE_EXCLUDE_KEYS}
    payload = {
        "backend": sanitized,
        "image": image_rel_path,
        "prompt": prompt,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


@dataclass
class TriageResult:
    image_path: str
    ground_truth: str
    ground_truth_triage: str
    model_prediction: str
    model_triage: str
    gated_prediction: Optional[str]
    gated_triage: Optional[str]
    quality_flags: Dict[str, float]
    quality_fail_labels: List[str]
    quality_fail: bool
    latency: float
    token_usage: Optional[int]


def _triage_results_to_records(results: List[TriageResult]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for r in results:
        record: Dict[str, Any] = {
            "image_path": r.image_path,
            "ground_truth": r.ground_truth,
            "ground_truth_triage": r.ground_truth_triage,
            "model_prediction": r.model_prediction,
            "model_triage": r.model_triage,
            "gated_prediction": r.gated_prediction,
            "gated_triage": r.gated_triage,
            "quality_fail": r.quality_fail,
            "quality_fail_labels": ";".join(r.quality_fail_labels),
            "latency": r.latency,
            "token_usage": r.token_usage,
        }
        for label in TARGET_LABELS:
            record[f"quality_score_{label}"] = r.quality_flags.get(label)
        records.append(record)
    return records


def build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])


class QualityGate:
    def __init__(self, checkpoint: Path, thresholds: Dict[str, float]) -> None:
        state = torch.load(checkpoint, map_location="cpu")
        config = state.get("config", {})
        self.model = QualityModel(
            architecture=config.get("architecture", "vit_small_patch16_224"),
            num_classes=len(TARGET_LABELS),
            pretrained=False,
        )
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.thresholds = thresholds
        self.preprocess = build_preprocess()

    def assess(self, image: Image.Image) -> Dict[str, float]:
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
        return {label: float(prob) for label, prob in zip(TARGET_LABELS, probs)}

    def failing_labels(self, scores: Dict[str, float]) -> List[str]:
        overall_key = 'overall_fail'
        fail_labels: List[str] = []

        if overall_key in scores:
            overall_threshold = self.thresholds.get(overall_key)
            if overall_threshold is not None and scores[overall_key] >= overall_threshold:
                fail_labels.append(overall_key)

        for label, prob in scores.items():
            if label == overall_key:
                continue
            threshold = self.thresholds.get(label)
            if threshold is None:
                threshold = 0.5
            if prob >= threshold:
                fail_labels.append(label)
        return fail_labels

    def should_retake(self, scores: Dict[str, float]) -> bool:
        return bool(self.failing_labels(scores))


def map_diagnosis_to_triage(diagnosis: str, triage_map: Dict[str, List[str]]) -> str:
    diag_lower = diagnosis.lower()
    for triage_label, keywords in triage_map.items():
        if any(keyword.lower() in diag_lower for keyword in keywords):
            return triage_label
    return "unknown"


def map_prediction_to_triage(prediction: str, triage_map: Dict[str, List[str]]) -> str:
    text = prediction.strip()
    candidate = prediction
    if text:
        block = text
        if text.startswith('```') and text.endswith('```'):
            stripped = text.strip('`')
            newline_index = stripped.find('\n')
            if newline_index != -1:
                block = stripped[newline_index + 1 :]
            else:
                block = stripped
        match = re.search(r'\{.*\}', block, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and 'diagnosis' in data:
                    value = data.get('diagnosis')
                    if isinstance(value, str):
                        candidate = value
                    elif isinstance(value, list) and value:
                        candidate = str(value[0])
            except json.JSONDecodeError:
                pass
    return map_diagnosis_to_triage(candidate, triage_map)



def compute_triage_metrics(results: List[TriageResult], triage_attr: str = "model_triage") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    y_true = [r.ground_truth_triage for r in results]
    normalized_predictions: List[str] = []
    retake_flags: List[bool] = []

    for r in results:
        value = getattr(r, triage_attr, None)
        if isinstance(value, str):
            candidate = value.strip().lower()
        elif value is None:
            candidate = ""
        else:
            candidate = str(value).strip().lower()

        quality_fail = bool(getattr(r, "quality_fail", False))

        if candidate in {"retake", "request_retake"}:
            label = "retake"
            is_retake = True
        elif candidate and candidate not in {"unknown", "n/a"}:
            label = candidate
            is_retake = quality_fail
        else:
            label = "unknown"
            is_retake = quality_fail

        normalized_predictions.append(label)
        retake_flags.append(is_retake)

    total_cases = len(normalized_predictions)
    total_known_truth = sum(1 for truth in y_true if truth != "unknown")

    valid_indices = [
        idx
        for idx, truth in enumerate(y_true)
        if truth != "unknown" and normalized_predictions[idx] not in {"unknown", "retake"}
    ]
    if valid_indices:
        correct = sum(1 for idx in valid_indices if y_true[idx] == normalized_predictions[idx])
        metrics["triage_accuracy"] = correct / len(valid_indices)
    metrics["triaged_cases"] = len(valid_indices)
    metrics["triaged_case_fraction"] = len(valid_indices) / max(1, total_known_truth)

    urgent_indices = [idx for idx in valid_indices if y_true[idx] == "urgent"]
    if urgent_indices:
        tp = sum(1 for idx in urgent_indices if normalized_predictions[idx] == "urgent")
        fn = sum(1 for idx in urgent_indices if normalized_predictions[idx] != "urgent")
        denom = tp + fn
        if denom:
            metrics["urgent_recall"] = tp / denom
            metrics["urgency_miss_rate"] = fn / denom

    total_urgent = sum(1 for truth in y_true if truth == "urgent")
    if total_urgent:
        urgent_deferrals = sum(
            1 for idx, truth in enumerate(y_true) if truth == "urgent" and retake_flags[idx]
        )
        metrics["urgent_deferral_rate"] = urgent_deferrals / total_urgent

    reassurance_indices = [idx for idx in valid_indices if y_true[idx] == "reassurance"]
    if reassurance_indices:
        safe = sum(1 for idx in reassurance_indices if normalized_predictions[idx] == "reassurance")
        metrics["safe_reassurance_rate"] = safe / len(reassurance_indices)

    metrics["retake_rate"] = sum(retake_flags) / max(1, total_cases)
    metrics["mean_latency"] = float(np.mean([r.latency for r in results])) if results else float("nan")
    token_usages = [r.token_usage for r in results if r.token_usage is not None]
    metrics["mean_token_usage"] = float(np.mean(token_usages)) if token_usages else float("nan")
    return metrics


def run_triage_simulation(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(config["input_manifests"]["baseline"])
    simulation_cfg = config.get("simulation", {})
    random_subset = simulation_cfg.get("random_subset", config.get("random_subset", False))
    num_images = simulation_cfg.get("num_images", config.get("num_images", len(manifest)))
    seed = simulation_cfg.get("seed", config.get("seed", 42))
    if random_subset:
        manifest = manifest.sample(n=num_images, random_state=seed)
    elif num_images and num_images < len(manifest):
        manifest = manifest.head(num_images)

    total_images = len(manifest)
    triage_labels = config.get("triage_labels", {})
    vlm_results: List[TriageResult] = []
    backend_summaries: List[Dict[str, float]] = []

    quality_gate: Optional[QualityGate] = None
    quality_cfg = config.get("quality_model")
    if quality_cfg and Path(quality_cfg["checkpoint"]).exists():
        quality_gate = QualityGate(Path(quality_cfg["checkpoint"]), quality_cfg.get("thresholds", {}))
    else:
        LOGGER.warning("Quality checkpoint missing; running without gating.")

    dataset_dir = Path(config.get("quality_dataset_dir", QUALITY_DATA_DIR))
    cache_responses = simulation_cfg.get("cache_responses", False)
    cache_root: Optional[Path] = None
    if cache_responses:
        cache_root = Path(simulation_cfg.get("cache_dir", "reports/triage/cache"))
        cache_root.mkdir(parents=True, exist_ok=True)

    store_traces = simulation_cfg.get("store_reasoning_traces", False)
    trace_root: Optional[Path] = None
    if store_traces:
        trace_root = Path(simulation_cfg.get("trace_dir", "reports/triage/traces"))
        trace_root.mkdir(parents=True, exist_ok=True)

    outputs_cfg = config.get("outputs", {})
    detail_target_value = outputs_cfg.get("summary_csv")
    if detail_target_value:
        detail_target = Path(detail_target_value)
    else:
        detail_target = REPORT_DIR / "triage" / "triage_detail.csv"
    summary_target_value = outputs_cfg.get("summary_backend_csv")
    if summary_target_value:
        summary_target = Path(summary_target_value)
    else:
        summary_target = REPORT_DIR / "triage" / "triage_summary.csv"
    partial_detail_value = outputs_cfg.get("partial_detail_csv")
    if partial_detail_value:
        partial_detail_path = Path(partial_detail_value)
    else:
        partial_detail_path = detail_target.with_name(detail_target.stem + ".partial" + detail_target.suffix)
    partial_summary_value = outputs_cfg.get("partial_summary_csv")
    if partial_summary_value:
        partial_summary_path = Path(partial_summary_value)
    else:
        partial_summary_path = summary_target.with_name(summary_target.stem + ".partial" + summary_target.suffix)
    progress_log_value = outputs_cfg.get("progress_log")
    if progress_log_value:
        progress_log_path = Path(progress_log_value)
    else:
        progress_log_path = detail_target.with_name(detail_target.stem + "_progress.log")
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text(
        f"{datetime.utcnow().isoformat()}Z | triage simulation started with {total_images} images across "
        f"{len(config.get('vlm_backends', []))} backend(s)\n",
        encoding="utf-8",
    )

    def log_progress(message: str) -> None:
        LOGGER.info(message)
        try:
            with progress_log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.utcnow().isoformat()}Z | {message}\n")
        except OSError as exc:
            LOGGER.warning("Failed to write progress log: %s", exc)

    for backend_cfg in config.get("vlm_backends", []):
        backend = load_vlm_backend(backend_cfg)
        backend_results: List[TriageResult] = []
        backend_label = backend_cfg.get("name") or backend_cfg.get("model") or backend_cfg.get("type") or "unknown"
        backend_cache_dir: Optional[Path] = None
        if cache_root is not None:
            backend_cache_dir = _prepare_cache_dir(cache_root, backend_cfg)
        backend_trace_dir: Optional[Path] = None
        if trace_root is not None:
            backend_trace_dir = _prepare_cache_dir(trace_root, backend_cfg)

        progress_every = max(1, total_images // 20) if total_images else 1
        log_progress(f"Starting backend {backend_label} ({total_images} images)")

        prompt = backend_cfg.get("prompt_template_text")
        if not prompt and backend_cfg.get("prompt_template"):
            prompt = Path(backend_cfg["prompt_template"]).read_text(encoding="utf-8")
        if not prompt:
            prompt = "Describe the lesion."

        for row_idx, (_, row) in enumerate(manifest.iterrows(), start=1):
            image_rel_path = str(row["image_path"])
            image_path = dataset_dir / image_rel_path
            image = Image.open(image_path).convert("RGB")

            response: Optional[VLMResponse] = None
            latency = 0.0
            cache_path: Optional[Path] = None
            cache_hit = False
            if backend_cache_dir is not None:
                cache_path = _cache_file_path(backend_cache_dir, backend_cfg, image_rel_path, prompt)
                if cache_path.exists():
                    try:
                        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                        response = VLMResponse(
                            text=cache_payload["text"],
                            raw=None,
                            latency=cache_payload.get("latency"),
                            token_usage=cache_payload.get("token_usage"),
                        )
                        latency = cache_payload.get("latency", 0.0)
                        cache_hit = True
                        LOGGER.debug("Cache hit for %s (%s)", image_rel_path, backend_label)
                    except (json.JSONDecodeError, KeyError) as exc:
                        LOGGER.warning(
                            "Failed to load cached response for %s (%s): %s",
                            image_rel_path,
                            backend_label,
                            exc,
                        )

            if response is None:
                start = time.perf_counter()
                response = backend.generate(image, prompt)
                latency = time.perf_counter() - start
                response.latency = latency
                if cache_path is not None:
                    cache_payload = {
                        "text": response.text,
                        "token_usage": response.token_usage,
                        "latency": latency,
                    }
                    try:
                        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
                    except OSError as exc:
                        LOGGER.warning(
                            "Failed to write cache for %s (%s): %s",
                            image_rel_path,
                            backend_label,
                            exc,
                        )
            else:
                response.latency = latency

            prediction = response.text

            if backend_trace_dir is not None:
                trace_path = _cache_file_path(backend_trace_dir, backend_cfg, image_rel_path, prompt)
                trace_payload = {
                    "image_path": image_rel_path,
                    "prompt": prompt,
                    "response_text": prediction,
                    "latency": response.latency,
                    "token_usage": response.token_usage,
                    "cache_hit": cache_hit,
                }
                try:
                    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8")
                except OSError as exc:
                    LOGGER.warning(
                        "Failed to write trace for %s (%s): %s",
                        image_rel_path,
                        backend_label,
                        exc,
                    )

            model_triage = map_prediction_to_triage(prediction, triage_labels)
            gt_triage = map_diagnosis_to_triage(row.get("diagnosis", "unknown"), triage_labels)

            quality_scores: Dict[str, float] = {}
            fail_labels: List[str] = []
            gated_prediction: Optional[str] = None
            gated_triage: Optional[str] = None
            quality_fail = False
            if quality_gate:
                quality_scores = quality_gate.assess(image)
                fail_labels = quality_gate.failing_labels(quality_scores)
                quality_fail = bool(fail_labels)
                if quality_fail:
                    # Simulate retake: load the "pass" version of the image and re-triage
                    pass_image_rel_path = image_rel_path.replace("_fail.jpg", "_pass.jpg")
                    pass_image_path = dataset_dir / pass_image_rel_path
                    if pass_image_path.exists():
                        pass_image = Image.open(pass_image_path).convert("RGB")
                        
                        # Check cache for the pass image
                        pass_cache_path: Optional[Path] = None
                        pass_cache_hit = False
                        pass_response: Optional[VLMResponse] = None
                        if backend_cache_dir is not None:
                            pass_cache_path = _cache_file_path(backend_cache_dir, backend_cfg, pass_image_rel_path, prompt)
                            if pass_cache_path.exists():
                                try:
                                    pass_cache_payload = json.loads(pass_cache_path.read_text(encoding="utf-8"))
                                    pass_response = VLMResponse(
                                        text=pass_cache_payload["text"],
                                        raw=None,
                                        latency=pass_cache_payload.get("latency"),
                                        token_usage=pass_cache_payload.get("token_usage"),
                                    )
                                    pass_cache_hit = True
                                    LOGGER.debug("Cache hit for pass image %s (%s)", pass_image_rel_path, backend_label)
                                except (json.JSONDecodeError, KeyError) as exc:
                                    LOGGER.warning(
                                        "Failed to load cached response for pass image %s (%s): %s",
                                        pass_image_rel_path,
                                        backend_label,
                                        exc,
                                    )

                        if pass_response is None:
                            start_pass = time.perf_counter()
                            pass_response = backend.generate(pass_image, prompt)
                            latency_pass = time.perf_counter() - start_pass
                            pass_response.latency = latency_pass
                            if pass_cache_path is not None:
                                pass_cache_payload = {
                                    "text": pass_response.text,
                                    "token_usage": pass_response.token_usage,
                                    "latency": latency_pass,
                                }
                                try:
                                    pass_cache_path.write_text(json.dumps(pass_cache_payload, ensure_ascii=False), encoding="utf-8")
                                except OSError as exc:
                                    LOGGER.warning(
                                        "Failed to write cache for pass image %s (%s): %s",
                                        pass_image_rel_path,
                                        backend_label,
                                        exc,
                                    )
                        else:
                            pass_response.latency = pass_response.latency if pass_response.latency is not None else 0.0

                        gated_prediction = pass_response.text
                        gated_triage = map_prediction_to_triage(pass_response.text, triage_labels)
                        latency += pass_response.latency # Add latency of retake
                        if response.token_usage is not None and pass_response.token_usage is not None:
                            response.token_usage += pass_response.token_usage # Add token usage of retake
                        elif pass_response.token_usage is not None:
                            response.token_usage = pass_response.token_usage
                    else:
                        LOGGER.warning("Pass image not found for %s; defaulting to retake.", image_rel_path)
                        gated_prediction = "request_retake"
                        gated_triage = "retake"
                else:
                    gated_prediction = prediction
                    gated_triage = model_triage
            else:
                fail_labels = []

            backend_results.append(
                TriageResult(
                    image_path=row["image_path"],
                    ground_truth=row.get("diagnosis", "unknown"),
                    ground_truth_triage=gt_triage,
                    model_prediction=prediction,
                    model_triage=model_triage,
                    gated_prediction=gated_prediction,
                    gated_triage=gated_triage,
                    quality_flags=quality_scores,
                    quality_fail_labels=fail_labels,
                    quality_fail=quality_fail,
                    latency=latency,
                    token_usage=response.token_usage,
                )
            )

            if total_images and (row_idx == 1 or row_idx % progress_every == 0 or row_idx == total_images):
                log_progress(
                    f"{backend_label}: processed {row_idx}/{total_images} images ({row_idx / total_images:.1%})"
                )

        baseline_metrics = compute_triage_metrics(backend_results, triage_attr="model_triage")
        backend_summaries.append({"backend": backend_label, "mode": "baseline", **baseline_metrics})
        LOGGER.info("Backend %s baseline metrics: %s", backend_label, baseline_metrics)
        log_progress(f"{backend_label}: baseline metrics {baseline_metrics}")

        if any(result.gated_triage is not None for result in backend_results):
            gated_metrics = compute_triage_metrics(backend_results, triage_attr="gated_triage")
            backend_summaries.append({"backend": backend_label, "mode": "quality_gated", **gated_metrics})
            LOGGER.info("Backend %s gated metrics: %s", backend_label, gated_metrics)
            log_progress(f"{backend_label}: quality gated metrics {gated_metrics}")

        vlm_results.extend(backend_results)

        if vlm_results:
            partial_detail_path.parent.mkdir(parents=True, exist_ok=True)
            detail_partial_df = pd.DataFrame(_triage_results_to_records(vlm_results))
            detail_partial_df.to_csv(partial_detail_path, index=False)
            log_progress(f"Progress snapshot saved: {partial_detail_path}")
        if backend_summaries:
            partial_summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(backend_summaries).to_csv(partial_summary_path, index=False)
            log_progress(f"Progress snapshot saved: {partial_summary_path}")

    detail_df = pd.DataFrame(_triage_results_to_records(vlm_results))
    summary_df = pd.DataFrame(backend_summaries)
    log_progress("Triage simulation complete")
    return detail_df, summary_df



__all__ = ["run_triage_simulation", "QualityGate", "TriageResult"]
