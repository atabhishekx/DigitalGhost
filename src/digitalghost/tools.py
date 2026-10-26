"""Read-only tools used by the DigitalGhost investigation orchestrator."""
from __future__ import annotations

import pandas as pd


def _identity_row(identity_id: str, analyzed_df: pd.DataFrame) -> dict:
    if "identity_id" not in analyzed_df.columns:
        raise ValueError("Analysis results are missing the identity_id field.")
    matches = analyzed_df[analyzed_df["identity_id"].astype(str) == str(identity_id)]
    if matches.empty:
        raise ValueError(f"Unknown identity_id: {identity_id}")
    return matches.iloc[0].to_dict()


def get_identity_context(identity_id: str, bundle: dict) -> dict:
    """Return identity and effective assignment context without performing writes."""
    identities = {str(item.get("identity_id")): item for item in bundle.get("identities", []) if item.get("identity_id") is not None}
    identity = identities.get(str(identity_id))
    if identity is None:
        raise ValueError(f"Unknown identity_id: {identity_id}")
    assignments = [item for item in bundle.get("assignments", []) if str(item.get("identity_id")) == str(identity_id)]
    permission_map = {str(item.get("permission_id")): item for item in bundle.get("permissions", []) if item.get("permission_id") is not None}
    role_ids = list(dict.fromkeys(str(item.get("role_id", "")) for item in assignments if item.get("role_id")))
    permission_ids = list(dict.fromkeys(str(pid) for item in assignments for pid in (item.get("permission_ids", []) if isinstance(item.get("permission_ids", []), list) else str(item.get("permission_ids", "")).split(",")) if str(pid).strip()))
    return {
        "identity_id": str(identity_id),
        "name": str(identity.get("name", identity_id)),
        "lifecycle_status": str(identity.get("lifecycle_status", "Unknown")),
        "department": str(identity.get("department", "Unknown")),
        "user_type": str(identity.get("user_type", "Unknown")),
        "assigned_role_ids": role_ids,
        "assigned_role_names": list(dict.fromkeys(str(item.get("role_name", item.get("role_id", "Unknown"))) for item in assignments)),
        "permission_ids": permission_ids,
        "permission_names": [str(permission_map.get(pid, {}).get("name", pid)) for pid in permission_ids],
        "permission_levels": [str(permission_map.get(pid, {}).get("privilege_level", "Unknown")) for pid in permission_ids],
        "source_type": str(identity.get("source_type", "unknown")),
    }


def get_rule_signal(identity_id: str, analyzed_df: pd.DataFrame) -> dict:
    row = _identity_row(identity_id, analyzed_df)
    return {"score": int(row.get("risk_score", 0)), "category": str(row.get("risk_category", "Unknown")),
            "reasons": list(row.get("reasons_list", []) or [])}


def get_graph_signal(identity_id: str, analyzed_df: pd.DataFrame) -> dict:
    row = _identity_row(identity_id, analyzed_df)
    return {"score": int(row.get("graph_risk_score", 0)),
            "high_privilege_count": int(row.get("graph_high_privilege_count", 0)),
            "sensitive_permission_count": int(row.get("graph_sensitive_permission_count", 0)),
            "sensitive_resource_count": int(row.get("graph_sensitive_resource_count", 0)),
            "max_permission_blast_radius": int(row.get("graph_max_permission_blast_radius", 0)),
            "reasons": list(row.get("graph_reasons", []) or [])}


def get_behavior_signal(identity_id: str, analyzed_df: pd.DataFrame) -> dict:
    row = _identity_row(identity_id, analyzed_df)
    return {"raw_score": float(row.get("anomaly_raw_score", 0)),
            "percentile": float(row.get("anomaly_percentile", 0)),
            "is_anomaly": bool(row.get("is_anomaly", False)),
            "note": "Relative anomaly signal, not a probability of compromise."}
