"""Combines independent risk signals without pretending they share a probability scale."""
from __future__ import annotations
import pandas as pd
from .risk import analyze_dataset
from .graph_analysis import calculate_graph_risk
from .ml import score_anomalies
from .fusion import combine_risk_signals, priority_band

def analyze_all(bundle: dict, weight_terminated: int = 45, weight_inactive: int = 30,
                weight_unused_high_priv: int = 25, inactive_days_threshold: int = 90,
                model_path=None) -> pd.DataFrame:
    from .data import bundle_to_frame
    df = bundle_to_frame(bundle)
    if df.empty: return df
    rules = analyze_dataset(df, weight_terminated, weight_inactive, weight_unused_high_priv, inactive_days_threshold)
    graph = calculate_graph_risk(bundle.get("identities", []), bundle.get("assignments", []), bundle.get("permissions", []))
    ml = score_anomalies(df, model_path=model_path)
    result = rules.merge(graph, on="identity_id", how="left", validate="one_to_one").join(ml)
    result["priority_score"] = combine_risk_signals(result["risk_score"], result["graph_risk_score"], result["anomaly_percentile"], result["is_anomaly"])
    result["priority_band"] = result["priority_score"].map(priority_band)
    result["priority_explanation"] = result.apply(lambda r: f"Rule {r['risk_category'].lower()}; graph score {int(r['graph_risk_score'])}/100; anomaly percentile {r['anomaly_percentile']:.1f}. The triage score includes signal floors so a strong independent signal is not averaged away; it is not an incident probability.", axis=1)
    return result.sort_values(["priority_score", "risk_score"], ascending=False).reset_index(drop=True)
