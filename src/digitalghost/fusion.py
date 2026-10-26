"""Transparent risk-signal fusion that prevents one strong signal being averaged away."""
from __future__ import annotations

import pandas as pd


def combine_risk_signals(rule_score, graph_score, anomaly_percentile, is_anomaly) -> pd.Series:
    """Return a triage score, not a probability.

    The weighted score provides a combined view, while signal floors preserve the
    importance of a strong independent rule, graph or flagged-anomaly result.
    """
    rule = pd.to_numeric(pd.Series(rule_score), errors="coerce").fillna(0).clip(0, 100)
    graph = pd.to_numeric(pd.Series(graph_score, index=rule.index), errors="coerce").fillna(0).clip(0, 100)
    anomaly = pd.to_numeric(pd.Series(anomaly_percentile, index=rule.index), errors="coerce").fillna(0).clip(0, 100)
    flagged = pd.Series(is_anomaly, index=rule.index).fillna(False).astype(bool)
    weighted = 0.45 * rule + 0.30 * graph + 0.25 * anomaly
    anomaly_floor = anomaly.where(flagged, anomaly * 0.25) * 0.75
    signal_floor = pd.concat([0.80 * rule, 0.75 * graph, anomaly_floor], axis=1).max(axis=1)
    return pd.concat([weighted, signal_floor], axis=1).max(axis=1).clip(0, 100).round(1)


def priority_band(score: float) -> str:
    value = float(score)
    if value >= 75:
        return "Urgent"
    if value >= 50:
        return "High"
    if value >= 25:
        return "Review"
    return "Routine"
