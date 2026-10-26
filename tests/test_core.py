import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pytest
from digitalghost.data import generate_synthetic_bundle, validate_bundle, bundle_to_frame, ensure_project_data, persist_bundle, verify_audit_chain, write_audit
from digitalghost.risk import evaluate_identity_risk, analyze_dataset
from digitalghost.graph_analysis import build_identity_graph, calculate_graph_risk
from digitalghost.ml import feature_frame, train_model, score_anomalies
from digitalghost.storage import write_json, read_json, append_jsonl, read_jsonl, atomic_write_bytes
from digitalghost.public_import import normalize_public_csv, save_public_import
from digitalghost.agent import investigate, record_analyst_decision
from digitalghost.analysis import analyze_all
from digitalghost.tools import get_identity_context, get_rule_signal, get_graph_signal, get_behavior_signal
from digitalghost.evaluation import evaluate_synthetic_bundle
from digitalghost.fusion import combine_risk_signals, priority_band


def test_risk_rules_and_boundaries():
    row = {"identity_id": "x", "lifecycle_status": "Terminated", "last_activity_days": 91,
           "privilege_level": "High", "permission_used_last_30d": False}
    score, category, reasons = evaluate_identity_risk(row)
    assert score == 100 and category == "Critical Risk" and len(reasons) == 3
    assert evaluate_identity_risk({"lifecycle_status": "Active", "last_activity_days": 90, "privilege_level": "Low"})[0] == 0
    with pytest.raises(ValueError): evaluate_identity_risk(row, weight_terminated=-1)


def test_generator_reproducible_shape_and_valid_references():
    a = generate_synthetic_bundle(80, events_per_identity=2, seed=9)
    b = generate_synthetic_bundle(80, events_per_identity=2, seed=9)
    assert len(a["identities"]) == 80 and len(a["events"]) == 160
    assert [x["identity_id"] for x in a["identities"]] == [x["identity_id"] for x in b["identities"]]
    assert validate_bundle(a) == []
    assert len({x["identity_id"] for x in a["identities"]}) == 80


def test_graph_uses_unique_ids_when_display_names_match():
    identities = [{"identity_id": "id:1", "name": "Same Name", "lifecycle_status": "Active"},
                  {"identity_id": "id:2", "name": "Same Name", "lifecycle_status": "Active"}]
    assignments = [{"identity_id": "id:1", "role_id": "role:x", "role_name": "Role", "permission_ids": ["p:x"]},
                   {"identity_id": "id:2", "role_id": "role:x", "role_name": "Role", "permission_ids": ["p:x"]}]
    g = build_identity_graph(identities, assignments, [{"permission_id": "p:x", "name": "s3:DeleteBucket", "privilege_level": "High"}])
    assert "identity::id:1" in g and "identity::id:2" in g
    assert g.number_of_nodes() >= 4
    result = calculate_graph_risk(identities, assignments, [{"permission_id": "p:x", "name": "s3:DeleteBucket", "privilege_level": "High"}])
    assert len(result) == 2


def test_model_excludes_scenario_labels_and_scores(tmp_path):
    bundle = generate_synthetic_bundle(100, events_per_identity=1, seed=4)
    df = bundle_to_frame(bundle)
    assert "scenario_label" not in feature_frame(df).columns
    path = tmp_path / "model.joblib"
    train_model(df, path)
    scores = score_anomalies(df, path, auto_train=False)
    assert len(scores) == len(df)
    assert scores["anomaly_percentile"].between(0, 100).all()
    assert scores["is_anomaly"].dtype == bool


def test_storage_atomic_json_and_jsonl(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"a": 1})
    write_json(path, {"a": 2}, backup=True)
    assert read_json(path)["a"] == 2
    assert read_json(path.with_suffix(".json.bak"))["a"] == 1
    log = tmp_path / "log.jsonl"
    append_jsonl(log, {"x": 1}); append_jsonl(log, {"x": 2})
    assert len(read_jsonl(log)) == 2


def test_public_csv_import_maps_fields_and_checks_user_column():
    content = b"user,timestamp,activity,url,label\nalex,2025-01-01T10:00:00Z,login,internal,anomaly\n"
    events, manifest = normalize_public_csv(content, "demo", "https://example.org/data", "CC BY")
    assert events[0]["identity_external_ref"] == "alex"
    assert events[0]["event_type"] == "login"
    assert events[0]["label_external"] == "anomaly"
    assert manifest["normalized_event_count"] == 1 and manifest["sha256"]
    with pytest.raises(ValueError): normalize_public_csv(b"foo,bar\n1,2\n", "bad")


def test_end_to_end_analysis_and_investigation_persistence(tmp_path):
    bundle = generate_synthetic_bundle(80, events_per_identity=1, seed=11)
    frame = bundle_to_frame(bundle)
    model_path = tmp_path / "model.joblib"
    train_model(frame, model_path)
    result = analyze_all(bundle, model_path=model_path)
    assert len(result) == 80
    assert {"risk_score", "graph_risk_score", "anomaly_percentile", "priority_score"}.issubset(result.columns)
    report = investigate(result.iloc[0]["identity_id"], result, root=tmp_path)
    assert report["investigation_id"].startswith("INV-")
    decision = record_analyst_decision(report["investigation_id"], "Needs review", "check source", root=tmp_path)
    assert decision["decision"] == "Needs review"
    assert (tmp_path / "data/state/audit.jsonl").exists()
    assert verify_audit_chain(tmp_path)["valid"] is True
    assert len(read_json(tmp_path / "data/state/investigations.json")) == 1


def test_multiple_assignments_are_aggregated_without_losing_unused_high_privilege():
    bundle = generate_synthetic_bundle(30, events_per_identity=1, seed=17)
    first = bundle["assignments"][0]
    extra = dict(first)
    extra["assignment_id"] = "asgn:extra"
    extra["role_id"] = "role:qa"
    extra["role_name"] = "QA Tester"
    extra["permission_ids"] = ["perm:app:ExecuteTests"]
    extra["permission_names"] = ["app:ExecuteTests"]
    extra["permission_levels"] = ["Low"]
    extra["resource_names"] = ["Test Environment"]
    extra["permission_used_last_30d"] = True
    bundle["assignments"].append(extra)
    frame = bundle_to_frame(bundle)
    assert len(frame) == 30
    row = frame.loc[frame.identity_id == first["identity_id"]].iloc[0]
    assert "QA Tester" in row.role_name
    assert len(row.permission_ids) >= 2
    assert str(row.unused_high_privilege).lower() in {"true", "false"}


def test_partial_existing_dataset_is_not_silently_overwritten(tmp_path):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    sentinel = proc / "identities.json"
    sentinel.write_text('[{"identity_id":"keep-me"}]', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="existing files were preserved"):
        ensure_project_data(tmp_path)
    assert json.loads(sentinel.read_text(encoding="utf-8"))[0]["identity_id"] == "keep-me"


def test_audit_chain_integrity_detects_tampering(tmp_path):
    write_audit(tmp_path, "test_one", {"value": 1})
    write_audit(tmp_path, "test_two", {"value": 2})
    assert verify_audit_chain(tmp_path)["valid"] is True
    audit_path = tmp_path / "data" / "state" / "audit.jsonl"
    records = read_jsonl(audit_path)
    records[0]["details"]["value"] = 99
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    result = verify_audit_chain(tmp_path)
    assert result["valid"] is False and result["broken_at"] == 1


def test_public_import_preserves_source_bytes_and_invalid_timestamps(tmp_path):
    content = b"user,timestamp,activity\r\nalex,not-a-date,login\r\n"
    events, manifest = normalize_public_csv(content, "byte test", "https://example.org", "CC BY")
    assert events[0]["timestamp"] is None
    outdir = save_public_import(tmp_path, content, events, manifest)
    assert (outdir / "source.csv").read_bytes() == content
    assert __import__("hashlib").sha256((outdir / "source.csv").read_bytes()).hexdigest() == manifest["sha256"]


def test_analyst_decision_rejects_unknown_investigation(tmp_path):
    with pytest.raises(ValueError, match="Unknown investigation_id"):
        record_analyst_decision("INV-does-not-exist", "Needs review", root=tmp_path)


def test_investigation_agent_calls_read_only_tools_and_records_trace(tmp_path):
    bundle = generate_synthetic_bundle(40, events_per_identity=1, seed=29)
    frame = bundle_to_frame(bundle)
    model_path = tmp_path / "models" / "model.joblib"
    train_model(frame, model_path)
    result = analyze_all(bundle, model_path=model_path)
    identity_id = result.iloc[0]["identity_id"]
    context = get_identity_context(identity_id, bundle)
    assert context["identity_id"] == identity_id
    assert get_rule_signal(identity_id, result)["score"] >= 0
    assert get_graph_signal(identity_id, result)["score"] >= 0
    assert 0 <= get_behavior_signal(identity_id, result)["percentile"] <= 100
    report = investigate(identity_id, result, bundle=bundle, root=tmp_path)
    assert len(report["tool_trace"]) == 4
    assert report["tool_trace"][0]["tool"] == "get_identity_context"


def test_corrupt_or_tampered_model_is_retrained_when_auto_train_enabled(tmp_path):
    bundle = generate_synthetic_bundle(40, events_per_identity=1, seed=31)
    frame = bundle_to_frame(bundle)
    model_path = tmp_path / "model.joblib"
    train_model(frame, model_path)
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    scores = score_anomalies(frame, model_path=model_path, auto_train=True)
    assert len(scores) == 40
    metadata = read_json(tmp_path / "model_metadata.json")
    assert metadata["artifact_sha256"] == __import__("hashlib").sha256(model_path.read_bytes()).hexdigest()


def test_bundle_validation_rejects_unknown_permission_reference():
    bundle = generate_synthetic_bundle(25, events_per_identity=1, seed=33)
    bundle["assignments"][0]["permission_ids"].append("perm:missing")
    assert any("unknown permission perm:missing" in error for error in validate_bundle(bundle))


def test_behavioral_features_match_generated_event_records():
    bundle = generate_synthetic_bundle(120, events_per_identity=8, seed=41)
    now_events = {}
    from datetime import datetime, timezone
    now = datetime.fromisoformat(bundle["manifest"]["created_at"])
    for event in bundle["events"]:
        timestamp = datetime.fromisoformat(event["timestamp"])
        if (now - timestamp).total_seconds() <= 7 * 86400:
            now_events.setdefault(event["identity_id"], []).append(event)
    for identity in bundle["identities"]:
        recent = now_events.get(identity["identity_id"], [])
        assert identity["failed_logins_7d"] == sum(event["event_type"] == "failed_login" for event in recent)
        assert identity["unique_resources_7d"] == len({event["resource_id"] for event in recent})
        assert identity["privileged_actions_7d"] == sum(event["event_type"] in {"privileged_action", "permission_change"} for event in recent)


def test_held_out_synthetic_evaluation_returns_comparable_metrics():
    bundle = generate_synthetic_bundle(160, events_per_identity=3, seed=43)
    report = evaluate_synthetic_bundle(bundle)
    assert report["train_identities"] + report["test_identities"] == 160
    assert report["dataset_type"] == "synthetic_only"
    assert {row["method"] for row in report["methods"]} == {
        "Isolation Forest", "Rule baseline (score ≥ 40)", "Graph baseline (score ≥ 50)", "Combined triage (score ≥ 50)"
    }
    assert all(0 <= row["precision"] <= 1 and 0 <= row["recall"] <= 1 for row in report["methods"])


def test_fusion_does_not_average_away_a_strong_anomaly_signal():
    score = combine_risk_signals([0], [0], [95], [True]).iloc[0]
    assert score >= 70
    assert priority_band(score) == "High"
    assert priority_band(combine_risk_signals([100], [0], [0], [False]).iloc[0]) == "Urgent"
