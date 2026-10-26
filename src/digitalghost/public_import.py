"""Conservative importer for public activity CSVs and labeled research benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import re
import pandas as pd

from .storage import atomic_write_text, atomic_write_bytes, write_json
from .data import write_audit

MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ROWS = 250_000


def _normalise_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_public_csv(
    content: bytes,
    source_name: str,
    source_url: str = "",
    license_text: str = "Verify source license before use.",
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[list[dict], dict]:
    """Normalize common event fields while preserving labels as external annotations.

    The importer deliberately does not join public identities to local synthetic identities,
    and does not treat a benchmark label as a permission-risk ground truth label.
    """
    if not isinstance(content, bytes):
        raise TypeError("CSV content must be bytes")
    if len(content) > MAX_BYTES:
        raise ValueError("CSV exceeds the 50 MB local-import limit.")
    if not content:
        raise ValueError("Uploaded CSV is empty.")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= DEFAULT_MAX_ROWS:
        raise ValueError(f"max_rows must be between 1 and {DEFAULT_MAX_ROWS:,}.")
    if not str(source_name).strip():
        raise ValueError("Dataset name is required for provenance.")

    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False, nrows=max_rows + 1)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("CSV contains no data rows.")
    if len(df) > max_rows:
        raise ValueError(f"CSV exceeds the {max_rows:,}-row import limit.")

    original_columns = list(df.columns)
    normalized_columns: dict[str, object] = {}
    for column in df.columns:
        key = _normalise_column(column)
        if key in normalized_columns:
            raise ValueError(f"Ambiguous CSV columns normalize to the same name: {normalized_columns[key]!r} and {column!r}.")
        normalized_columns[key] = column

    def find(*names: str):
        for name in names:
            if _normalise_column(name) in normalized_columns:
                return normalized_columns[_normalise_column(name)]
        return None

    user_col = find("user", "username", "user_id", "userid", "employee", "identity", "account", "src_user",
                    "case_id", "case id", "actor", "principal", "user name", "case:concept:name")
    time_col = find("date", "timestamp", "time", "datetime", "logon_time", "event_time", "time:timestamp")
    event_col = find("activity", "event_type", "action", "type", "activity_type", "operation", "concept:name")
    resource_col = find("url", "filename", "file", "resource", "host", "device", "destination", "org:resource")
    label_col = find("label", "class", "target", "ground_truth", "ground truth", "is_anomaly", "anomaly_label", "insider")
    category_col = find("category", "subcategory", "attack_type", "anomaly_type")
    if user_col is None:
        raise ValueError("Could not identify an identity/case column. Expected user, username, user_id, case_id, actor or a similar field.")

    imported_at = datetime.now(timezone.utc).isoformat()
    events = []
    for idx, row in df.iterrows():
        user_value = row.get(user_col)
        user = "" if pd.isna(user_value) else str(user_value).strip()
        if not user or user.lower() in {"nan", "none", "null"}:
            continue
        timestamp = None
        if time_col:
            parsed = pd.to_datetime(row.get(time_col), errors="coerce", utc=True)
            if not pd.isna(parsed):
                timestamp = parsed.isoformat()
        event_value = row.get(event_col, "activity") if event_col else "activity"
        resource_value = row.get(resource_col, "unknown") if resource_col else "unknown"
        label_value = row.get(label_col) if label_col else None
        category_value = row.get(category_col) if category_col else None
        events.append({
            "event_id": f"public:{idx + 1}",
            "identity_external_ref": user[:250],
            "timestamp": timestamp,
            "event_type": "activity" if pd.isna(event_value) else str(event_value)[:120],
            "resource_external_ref": "unknown" if pd.isna(resource_value) else str(resource_value)[:250],
            "label_external": None if label_col is None or pd.isna(label_value) else str(label_value)[:120],
            "category_external": None if category_col is None or pd.isna(category_value) else str(category_value)[:120],
            "source_type": "public_import",
            "source_name": str(source_name).strip()[:200],
            "success": None,
            "off_hours": None,
        })

    source_hash = hashlib.sha256(content).hexdigest()
    manifest = {
        "dataset_id": "public-" + source_hash[:16],
        "source_name": str(source_name).strip()[:200],
        "source_url": str(source_url).strip()[:1000],
        "license": str(license_text).strip()[:500],
        "source_type": "public_research_import",
        "imported_at": imported_at,
        "sha256": source_hash,
        "raw_bytes": len(content),
        "raw_rows": len(df),
        "normalized_event_count": len(events),
        "columns": [str(column) for column in original_columns],
        "mapping": {
            "identity_column": str(user_col),
            "timestamp_column": str(time_col) if time_col is not None else None,
            "event_column": str(event_col) if event_col is not None else None,
            "resource_column": str(resource_col) if resource_col is not None else None,
            "label_column": str(label_col) if label_col is not None else None,
            "category_column": str(category_col) if category_col is not None else None,
        },
        "label_note": "Imported labels are retained as external annotations only. They are not used as model features or assumed to represent excessive-permission ground truth.",
        "warning": "External identity references are not automatically merged with synthetic identities. Review the source schema, license and label meaning before evaluation.",
    }
    return events, manifest


def save_public_import(root: Path, content: bytes, events: list[dict], manifest: dict) -> Path:
    import_dir = root / "data" / "raw" / "public" / manifest["dataset_id"]
    existing_manifest_path = import_dir / "manifest.json"
    if existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        same_provenance = all(existing.get(field) == manifest.get(field) for field in ("source_name", "source_url", "license"))
        if existing.get("sha256") == manifest.get("sha256") and same_provenance:
            return import_dir
        raise FileExistsError("This content hash already exists with different source metadata. Review the existing import instead of overwriting provenance.")
    import_dir.mkdir(parents=True, exist_ok=True)
    # Preserve exact uploaded bytes so the recorded SHA-256 can be verified later.
    atomic_write_bytes(import_dir / "source.csv", content)
    atomic_write_text(import_dir / "normalized_events.jsonl", "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events))
    write_json(import_dir / "manifest.json", manifest)
    write_audit(root, "public_dataset_imported", {"dataset_id": manifest["dataset_id"], "event_count": len(events), "sha256": manifest["sha256"]})
    return import_dir
