from snapcheck.triage import TriageResult, compute_triage_metrics


def test_triage_metrics_handles_retake():
    results = [
        TriageResult(
            image_path="img",
            ground_truth="melanoma",
            ground_truth_triage="urgent",
            model_prediction="melanoma",
            model_triage="urgent",
            gated_prediction="request_retake",
            gated_triage="retake",
            quality_flags={},
            quality_fail=True,
            latency=0.5,
            token_usage=10,
        ),
        TriageResult(
            image_path="img2",
            ground_truth="nevus",
            ground_truth_triage="routine",
            model_prediction="nevus",
            model_triage="routine",
            gated_prediction="nevus",
            gated_triage="routine",
            quality_flags={},
            quality_fail=False,
            latency=0.4,
            token_usage=12,
        ),
    ]
    baseline = compute_triage_metrics(results, triage_attr="model_triage")
    gated = compute_triage_metrics(results, triage_attr="gated_triage")
    assert baseline["triage_accuracy"] == 1.0
    assert gated["retake_rate"] > 0
