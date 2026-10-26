"""Held-out evaluation helpers for the synthetic behavioral anomaly scenarios."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .data import bundle_to_frame, write_audit
from .graph_analysis import calculate_graph_risk
from .ml import score_anomalies, train_model
from .risk import analyze_dataset
from .fusion import combine_risk_signals
from .storage import read_json, write_json

POSITIVE_SCENARIOS = {"behavioral_spike", "failed_login_burst"}


def _metric_row(name: str, y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "method": name,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def evaluate_synthetic_bundle(bundle: dict) -> dict:
    """Compare behavioral anomaly detection against rules/graph on held-out identities.

    Labels are used only to score the evaluation; `feature_frame` explicitly excludes them.
    The benchmark is synthetic and is not a real-world performance estimate.
    """
    frame = bundle_to_frame(bundle)
    if frame.empty or "scenario_label" not in frame.columns:
        raise ValueError("Synthetic evaluation requires a non-empty bundle with scenario_label annotations.")
    if len(frame) < 80:
        raise ValueError("At least 80 identities are recommended for a stratified held-out evaluation.")
    labels = frame["scenario_label"].astype(str)
    if labels.nunique() < 2:
        raise ValueError("Synthetic evaluation needs at least two scenario classes.")
    positives = labels.isin(POSITIVE_SCENARIOS).sum()
    if positives < 2 or (len(labels) - positives) < 2:
        raise ValueError("Synthetic evaluation needs at least two positive and two negative behavior scenarios.")
    stratify_labels = labels if labels.value_counts().min() >= 2 else None
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=stratify_labels)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y_true = test["scenario_label"].astype(str).isin(POSITIVE_SCENARIOS).to_numpy()

    with tempfile.TemporaryDirectory(prefix="digitalghost-eval-") as temp_dir:
        model_path = Path(temp_dir) / "evaluation_model.joblib"
        train_model(train, model_path)
        ml_scores = score_anomalies(test, model_path=model_path, auto_train=False)

    rule_result = analyze_dataset(test)
    graph_result = calculate_graph_risk(bundle.get("identities", []), bundle.get("assignments", []), bundle.get("permissions", []))
    result = rule_result.merge(graph_result, on="identity_id", how="left", validate="one_to_one").join(ml_scores)
    priority = combine_risk_signals(result["risk_score"], result["graph_risk_score"], result["anomaly_percentile"], result["is_anomaly"])

    methods = [
        _metric_row("Isolation Forest", y_true, result["is_anomaly"].astype(bool).to_numpy()),
        _metric_row("Rule baseline (score ≥ 40)", y_true, (result["risk_score"] >= 40).to_numpy()),
        _metric_row("Graph baseline (score ≥ 50)", y_true, (result["graph_risk_score"] >= 50).to_numpy()),
        _metric_row("Combined triage (score ≥ 50)", y_true, (priority >= 50).to_numpy()),
    ]
    return {
        "evaluation_id": "EVAL-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": bundle.get("manifest", {}).get("dataset_id", "unknown"),
        "dataset_type": "synthetic_only",
        "total_identities": int(len(frame)),
        "train_identities": int(len(train)),
        "test_identities": int(len(test)),
        "positive_test_cases": int(y_true.sum()),
        "positive_label_definition": sorted(POSITIVE_SCENARIOS),
        "methods": methods,
        "limitations": "Synthetic held-out benchmark only. Results measure recovery of the generator's behavioral-spike and failed-login scenarios; they do not establish real-world attack detection performance.",
    }


def save_evaluation_report(root: Path, report: dict) -> None:
    path = root / "data" / "state" / "evaluation_reports.json"
    reports = read_json(path, []) or []
    reports.append(report)
    write_json(path, reports, backup=True)
    write_audit(root, "synthetic_evaluation_completed", {"evaluation_id": report["evaluation_id"], "dataset_id": report["dataset_id"], "test_identities": report["test_identities"]})
