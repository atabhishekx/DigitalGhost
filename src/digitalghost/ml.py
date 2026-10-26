"""Optional, locally persisted Isolation Forest anomaly model."""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .storage import write_json, read_json


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

NUMERIC = [
    "last_activity_days",
    "failed_logins_7d",
    "unique_resources_7d",
    "privileged_actions_7d",
    "off_hours_events_7d",
    "baseline_activity",
    "permission_count",
]

CATEGORICAL = [
    "lifecycle_status",
    "user_type",
    "department",
    "privilege_level",
]

FEATURES = NUMERIC + CATEGORICAL


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare only approved model features.

    Scenario labels, risk labels, IDs, names and outcomes are excluded.
    Missing numeric fields become zero; missing categorical fields become
    'Unknown'.
    """
    result = pd.DataFrame(index=df.index)

    for col in NUMERIC:
        source = (
            df[col]
            if col in df.columns
            else pd.Series(0, index=df.index)
        )
        result[col] = (
            pd.to_numeric(source, errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )

    for col in CATEGORICAL:
        source = (
            df[col]
            if col in df.columns
            else pd.Series("Unknown", index=df.index)
        )
        result[col] = source.fillna("Unknown").astype(str)

    return result[FEATURES]


def _build_model(contamination: float = 0.10) -> Pipeline:
    """Build the preprocessing and anomaly-detection pipeline."""
    transformer = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    detector = IsolationForest(
        n_estimators=180,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("features", transformer),
        ("detector", detector),
    ])


def train_model(
    df: pd.DataFrame,
    model_path: Path | None = None,
) -> dict:
    """Train and atomically save the Isolation Forest model.

    A temporary model file is created first. The existing artifact is
    replaced only after serialization and file synchronization succeed.
    """
    if len(df) < 20:
        raise ValueError(
            "At least 20 identities are required to train the anomaly model."
        )

    model_path = model_path or MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X = feature_frame(df)

    model = _build_model()
    model.fit(X)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{model_path.name}.",
        suffix=".tmp",
        dir=str(model_path.parent),
    )
    os.close(fd)

    try:
        joblib.dump(model, temp_name)

        # Use a writable file descriptor for fsync.
        # Opening with "rb" can cause a bad-file-descriptor error on Windows.
        with open(temp_name, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        # Atomic replacement prevents a partially written model artifact.
        os.replace(temp_name, model_path)

    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    feature_digest = hashlib.sha256(
        X.to_csv(index=False).encode("utf-8")
    ).hexdigest()

    artifact_digest = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    metadata = {
        "model_type": "IsolationForest",
        "model_version": "1.1",
        "trained_rows": len(df),
        "features": FEATURES,
        "feature_data_sha256": feature_digest,
        "artifact_sha256": artifact_digest,
        "contamination": 0.10,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_note": (
            "Unsupervised anomaly detector. "
            "Synthetic scenario labels are not model features."
        ),
        "evaluation_note": (
            "Scores are relative anomaly signals, not calibrated "
            "probabilities or proof of malicious activity."
        ),
    }

    write_json(
        model_path.parent / "model_metadata.json",
        metadata,
        backup=True,
    )

    return metadata


def _load_model_verified(model_path: Path):
    """Verify the saved model checksum before loading it."""
    metadata = read_json(
        model_path.parent / "model_metadata.json",
        {},
    ) or {}

    expected_digest = (
        metadata.get("artifact_sha256")
        if isinstance(metadata, dict)
        else None
    )

    if expected_digest:
        actual_digest = hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest()

        if actual_digest != expected_digest:
            raise ValueError(
                "Saved anomaly model checksum does not match "
                "model_metadata.json."
            )

    return joblib.load(model_path)


def score_anomalies(
    df: pd.DataFrame,
    model_path: Path | None = None,
    auto_train: bool = True,
) -> pd.DataFrame:
    """Return raw anomaly scores, percentiles and anomaly flags."""
    model_path = model_path or MODEL_PATH

    if len(df) == 0:
        return pd.DataFrame(
            index=df.index,
            columns=[
                "anomaly_raw_score",
                "anomaly_percentile",
                "is_anomaly",
            ],
        )

    if not model_path.exists():
        if not auto_train:
            raise FileNotFoundError(
                "Anomaly model not found. Train the model first."
            )

        train_model(df, model_path)

    X = feature_frame(df)

    try:
        model = _load_model_verified(model_path)

        # Higher raw values indicate more anomalous records.
        raw = -model.decision_function(X)
        predictions = model.predict(X)

    except Exception as exc:
        if not auto_train:
            raise RuntimeError(
                f"Could not load or use the saved anomaly model "
                f"at {model_path}: {exc}"
            ) from exc

        # Recover from a corrupt or incompatible local model.
        train_model(df, model_path)

        model = _load_model_verified(model_path)
        raw = -model.decision_function(X)
        predictions = model.predict(X)

    ranks = (
        pd.Series(raw, index=df.index)
        .rank(pct=True, method="average")
        * 100
    )

    return pd.DataFrame(
        {
            "anomaly_raw_score": raw,
            "anomaly_percentile": ranks.round(1),
            "is_anomaly": predictions == -1,
        },
        index=df.index,
    )