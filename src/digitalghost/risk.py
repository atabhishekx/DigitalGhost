"""Transparent rule-based identity risk scoring."""
from __future__ import annotations
from typing import Any
import pandas as pd

CATEGORIES = ["Critical Risk", "Moderate Risk", "Low Risk", "Minimal Risk"]

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value): return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default

def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool): return value
    if value is None: return default
    if isinstance(value, (int, float)): return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def validate_weights(weight_terminated: int, weight_inactive: int, weight_unused_high_priv: int, inactive_days_threshold: int) -> None:
    for name, value in [("weight_terminated", weight_terminated), ("weight_inactive", weight_inactive), ("weight_unused_high_priv", weight_unused_high_priv)]:
        if not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError(f"{name} must be an integer between 0 and 100")
    if not isinstance(inactive_days_threshold, int) or not 1 <= inactive_days_threshold <= 3650:
        raise ValueError("inactive_days_threshold must be an integer between 1 and 3650")

def evaluate_identity_risk(row: Any, weight_terminated: int = 45, weight_inactive: int = 30,
                           weight_unused_high_priv: int = 25, inactive_days_threshold: int = 90) -> tuple[int, str, list[str]]:
    validate_weights(weight_terminated, weight_inactive, weight_unused_high_priv, inactive_days_threshold)
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    score, reasons = 0, []
    lifecycle = str(get("lifecycle_status", "Unknown") or "Unknown").strip().lower()
    if lifecycle in {"terminated", "suspended"}:
        score += weight_terminated
        reasons.append(f"Identity lifecycle is {lifecycle}; assigned access should be reviewed.")
    last_activity = _safe_int(get("last_activity_days", 0), 0)
    if last_activity > inactive_days_threshold:
        score += weight_inactive
        reasons.append(f"No recorded activity for {last_activity} days (configured threshold: {inactive_days_threshold}).")
    levels = get("permission_levels", [])
    if isinstance(levels, str): levels = [v.strip() for v in levels.split(",") if v.strip()]
    privilege = str(get("privilege_level", "Low") or "Low").strip().lower()
    has_high = privilege == "high" or (isinstance(levels, (list, tuple, set)) and any(str(x).lower() == "high" for x in levels))
    # Aggregated identities can have several assignments. Prefer the precise derived
    # indicator so a recently used high-privilege role is not confused with an unused low role.
    has_unused_high = _safe_bool(get("unused_high_privilege", None), False)
    try:
        has_explicit_unused_flag = "unused_high_privilege" in row
    except TypeError:
        has_explicit_unused_flag = False
    if has_explicit_unused_flag:
        trigger_unused_high = has_unused_high
    else:
        trigger_unused_high = has_high and not _safe_bool(get("permission_used_last_30d", False), False)
    if trigger_unused_high:
        score += weight_unused_high_priv
        reasons.append("At least one high-privilege assignment is recorded as unused in the last 30 days.")
    score = min(100, max(0, int(score)))
    category = "Critical Risk" if score >= 70 else "Moderate Risk" if score >= 40 else "Low Risk" if score > 0 else "Minimal Risk"
    return score, category, reasons

def analyze_dataset(df: pd.DataFrame, weight_terminated: int = 45, weight_inactive: int = 30,
                    weight_unused_high_priv: int = 25, inactive_days_threshold: int = 90) -> pd.DataFrame:
    validate_weights(weight_terminated, weight_inactive, weight_unused_high_priv, inactive_days_threshold)
    results = []
    for _, row in df.iterrows():
        score, category, reasons = evaluate_identity_risk(row, weight_terminated, weight_inactive, weight_unused_high_priv, inactive_days_threshold)
        results.append({"risk_score": score, "risk_category": category, "reasons_list": reasons,
                        "flagged_reasons": " | ".join(reasons) if reasons else "No configured rule triggered."})
    out = df.reset_index(drop=True).copy()
    for col in ["risk_score", "risk_category", "reasons_list", "flagged_reasons"]:
        out[col] = [r[col] for r in results]
    return out
