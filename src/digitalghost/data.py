"""Synthetic data creation, validation, and local data loading."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib, json, random, uuid
import pandas as pd
from .storage import read_json, write_json, append_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_SYNTH = DATA / "raw" / "synthetic"
PROCESSED = DATA / "processed"
STATE = DATA / "state"

FIRST = ["Alex", "Priya", "Jordan", "Taylor", "Rohan", "Elena", "Sam", "Chris", "Morgan", "Avery", "Nina", "Arjun", "Maya", "Noah", "Isha", "Kai", "Leah", "Dev", "Sofia", "Omar"]
LAST = ["Mercer", "Sharma", "Lee", "Patel", "Mehta", "Rostova", "Wilson", "Evans", "Morgan", "Brooks", "Rao", "Chen", "Singh", "Khan", "Reed", "Kapoor", "Das", "Turner", "Diaz", "Kim"]
ROLES = [
    ("role:cloud-admin", "Cloud Administrator", "High"),
    ("role:db-admin", "Database Administrator", "High"),
    ("role:devops", "DevOps Engineer", "High"),
    ("role:security", "Security Analyst", "High"),
    ("role:data-analyst", "Data Analyst", "Medium"),
    ("role:qa", "QA Tester", "Low"),
    ("role:finance", "Finance Analyst", "Medium"),
    ("role:hr", "HR Operations", "Medium"),
    ("role:contractor", "External Contractor", "Low"),
    ("role:service", "Service Account", "High"),
]
PERMS = [
    ("perm:iam:PassRole", "iam:PassRole", "High", "Cloud IAM"),
    ("perm:s3:DeleteBucket", "s3:DeleteBucket", "High", "Object Storage"),
    ("perm:db:DropDatabase", "db:DropDatabase", "High", "Production Database"),
    ("perm:kms:DecryptAll", "kms:DecryptAll", "High", "Encryption Keys"),
    ("perm:ec2:TerminateInstances", "ec2:TerminateInstances", "High", "Compute"),
    ("perm:db:ReadTable", "db:ReadTable", "Medium", "Analytics Database"),
    ("perm:app:ExecuteTests", "app:ExecuteTests", "Low", "Test Environment"),
    ("perm:doc:ReadInternalDocs", "doc:ReadInternalDocs", "Low", "Internal Documents"),
    ("perm:billing:View", "billing:View", "Low", "Billing"),
    ("perm:logs:Read", "logs:Read", "Medium", "Audit Logs"),
    ("perm:iam:CreateUser", "iam:CreateUser", "High", "Cloud IAM"),
    ("perm:storage:ReadSensitive", "storage:ReadSensitive", "High", "Sensitive Storage"),
    ("perm:queue:Publish", "queue:Publish", "Medium", "Message Queue"),
    ("perm:report:Export", "report:Export", "Medium", "Reporting"),
]

def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def generate_synthetic_bundle(n: int = 1500, events_per_identity: int = 8, seed: int = 20261026) -> dict:
    """Generate reproducible identities, assignments, and activity events.

    Scenario labels are retained only for evaluation; never included in model features.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 20 or n > 100_000:
        raise ValueError("n must be an integer between 20 and 100000")
    if isinstance(events_per_identity, bool) or not isinstance(events_per_identity, int) or not 1 <= events_per_identity <= 100:
        raise ValueError("events_per_identity must be an integer between 1 and 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be an integer between 0 and 2147483647")
    if n * events_per_identity > 1_000_000:
        raise ValueError("The generated dataset may contain at most 1,000,000 events per run.")
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    identities, assignments, events = [], [], []
    for i in range(1, n + 1):
        scenario = rng.choices(["normal", "terminated_access", "dormant_privileged", "behavioral_spike", "failed_login_burst", "overprivileged"], [0.70, .06, .08, .06, .04, .06])[0]
        status = "Terminated" if scenario == "terminated_access" else ("Suspended" if rng.random() < .025 else "Active")
        if scenario == "terminated_access": status = "Terminated"
        if scenario == "dormant_privileged":
            last_days = rng.randint(100, 240)
        elif scenario in {"behavioral_spike", "failed_login_burst"}:
            last_days = rng.randint(0, 2)
        else:
            last_days = rng.randint(0, 25)
        if status == "Terminated":
            last_days = rng.randint(5, 180)
        role = rng.choice(ROLES)
        if scenario in ("terminated_access", "dormant_privileged", "overprivileged"):
            role = rng.choice(ROLES[:4])
        identity_id = f"id:{seed}:{i:06d}"
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"  # duplicates intentionally possible; IDs remain unique
        identities.append({
            "identity_id": identity_id, "name": name,
            "user_type": rng.choice(["Employee", "Contractor", "Service Account"]),
            "lifecycle_status": status, "department": rng.choice(["Engineering", "Finance", "People", "Operations", "Data", "Security"]),
            "last_activity_days": last_days, "failed_logins_7d": 0,
            "unique_resources_7d": 0, "privileged_actions_7d": 0,
            "off_hours_events_7d": 0, "baseline_activity": round(rng.uniform(1, 12), 2),
            "scenario_label": scenario, "source_type": "digitalghost_synthetic", "created_at": now.isoformat()
        })
        perm_count = rng.randint(3, 7) if scenario == "overprivileged" else rng.randint(1, 3)
        candidate_perms = PERMS[:]
        if role[2] == "Low": candidate_perms = PERMS[5:10]
        chosen = rng.sample(candidate_perms, min(perm_count, len(candidate_perms)))
        if scenario in ("terminated_access", "dormant_privileged", "overprivileged") and not any(p[2] == "High" for p in chosen):
            chosen[0] = rng.choice(PERMS[:5])
        used_recently = not (scenario in ("terminated_access", "dormant_privileged") or rng.random() < .18)
        assignments.append({"assignment_id": f"asgn:{seed}:{i:06d}", "identity_id": identity_id,
                            "role_id": role[0], "role_name": role[1], "role_privilege": role[2],
                            "permission_ids": [p[0] for p in chosen], "permission_names": [p[1] for p in chosen],
                            "permission_levels": [p[2] for p in chosen], "resource_names": [p[3] for p in chosen],
                            "permission_used_last_30d": bool(used_recently), "active": status == "Active",
                            "source_type": "digitalghost_synthetic"})
        for j in range(events_per_identity):
            # Event timestamps are consistent with the identity's latest-activity age.
            if j == 0:
                age_days = last_days
            elif scenario in {"behavioral_spike", "failed_login_burst"}:
                age_days = last_days + rng.randint(0, 5)
            else:
                age_days = last_days + rng.randint(1, 90)
            if scenario == "behavioral_spike":
                ts = now - timedelta(days=age_days)
                ts = ts.replace(hour=rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23]), minute=rng.randint(0, 59), second=0, microsecond=0)
                if ts > now:
                    ts -= timedelta(days=1)
                event_type = rng.choice(["privileged_action", "resource_access", "permission_change"])
            elif scenario == "failed_login_burst":
                ts = now - timedelta(days=age_days, hours=rng.randint(0, 12))
                event_type = "failed_login"
            else:
                ts = now - timedelta(days=age_days, hours=rng.randint(0, 23))
                event_type = rng.choices(
                    ["login", "resource_access", "privileged_action", "file_read", "failed_login"],
                    [0.30, 0.30, 0.18, 0.18, 0.04],
                )[0]
            if scenario in {"behavioral_spike", "overprivileged"}:
                resource_id = f"resource:{rng.randint(1, max(12, events_per_identity * 3)):04d}"
            else:
                resource_id = rng.choice([p[0] for p in chosen])
            events.append({"event_id": f"evt:{seed}:{i:06d}:{j:03d}", "identity_id": identity_id,
                           "timestamp": ts.isoformat(), "event_type": event_type,
                           "resource_id": resource_id, "success": event_type != "failed_login",
                           "off_hours": ts.hour < 6 or ts.hour > 20, "source_type": "digitalghost_synthetic"})

    # Derive behavioral features from the event log instead of independently inventing
    # summary counts. This makes the feature table auditable against source events.
    events_by_identity: dict[str, list[dict]] = {}
    for event in events:
        events_by_identity.setdefault(event["identity_id"], []).append(event)
    for identity in identities:
        identity_events = events_by_identity.get(identity["identity_id"], [])
        parsed_events = [(datetime.fromisoformat(event["timestamp"]), event) for event in identity_events]
        if parsed_events:
            latest = max(timestamp for timestamp, _ in parsed_events)
            identity["last_activity_days"] = max(0, (now - latest).days)
        recent = [(timestamp, event) for timestamp, event in parsed_events if (now - timestamp).total_seconds() <= 7 * 86400]
        identity["failed_logins_7d"] = sum(event["event_type"] == "failed_login" for _, event in recent)
        identity["unique_resources_7d"] = len({event["resource_id"] for _, event in recent})
        identity["privileged_actions_7d"] = sum(event["event_type"] in {"privileged_action", "permission_change"} for _, event in recent)
        identity["off_hours_events_7d"] = sum(bool(event["off_hours"]) for _, event in recent)

    roles = [{"role_id": r[0], "name": r[1], "privilege_level": r[2]} for r in ROLES]
    permissions = [{"permission_id": p[0], "name": p[1], "privilege_level": p[2], "resource_name": p[3]} for p in PERMS]
    return {"identities": identities, "assignments": assignments, "events": events, "roles": roles, "permissions": permissions,
            "manifest": {"dataset_id": f"dg-synthetic-{seed}-{n}", "source_name": "DigitalGhost synthetic generator", "source_url": None,
                         "license": "Generated locally; no external data", "source_type": "synthetic", "generator_version": "2.1.0",
                         "seed": seed, "identity_count": n, "event_count": len(events), "created_at": now.isoformat(),
                         "label_note": "scenario_label is synthetic ground truth for evaluation only; exclude it from features."}}

def validate_bundle(bundle: dict) -> list[str]:
    """Validate IDs and references before analysis or persistence."""
    errors: list[str] = []
    if not isinstance(bundle, dict):
        return ["Dataset bundle must be a dictionary."]
    collection_names = ("identities", "assignments", "events", "roles", "permissions")
    collections = {}
    for name in collection_names:
        value = bundle.get(name, [])
        if not isinstance(value, list):
            errors.append(f"{name} must be a list of records.")
            value = []
        if any(not isinstance(row, dict) for row in value):
            errors.append(f"Every record in {name} must be an object/dictionary.")
            value = [row for row in value if isinstance(row, dict)]
        collections[name] = value
    identities = collections["identities"]
    assignments = collections["assignments"]
    events = collections["events"]
    roles = collections["roles"]
    permissions = collections["permissions"]
    if not isinstance(bundle.get("manifest", {}), dict):
        errors.append("Dataset manifest must be an object/dictionary.")

    def unique_ids(rows, key, label, *, required_table=False):
        if required_table and not rows:
            errors.append(f"{label} table must not be empty.")
        values = [row.get(key) if isinstance(row, dict) else None for row in rows]
        valid = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} records must have non-empty string {key} values.")
            else:
                valid.append(value)
        if len(valid) != len(set(valid)):
            errors.append(f"{key} values must be unique.")
        return set(valid)

    identity_ids = unique_ids(identities, "identity_id", "Identity", required_table=True)
    unique_ids(assignments, "assignment_id", "Assignment")
    unique_ids(events, "event_id", "Event")
    role_ids = unique_ids(roles, "role_id", "Role", required_table=True)
    permission_ids = unique_ids(permissions, "permission_id", "Permission", required_table=True)

    for row in identities:
        days = row.get("last_activity_days")
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            errors.append(f"Invalid last_activity_days for {row.get('identity_id')}")
        if row.get("lifecycle_status") not in {"Active", "Suspended", "Terminated"}:
            errors.append(f"Invalid lifecycle_status for {row.get('identity_id')}")

    for assignment in assignments:
        iid = assignment.get("identity_id")
        if iid not in identity_ids:
            errors.append(f"Assignment references unknown identity {iid}")
        role_id = assignment.get("role_id")
        if role_ids and role_id not in role_ids:
            errors.append(f"Assignment references unknown role {role_id}")
        pids = assignment.get("permission_ids", [])
        if isinstance(pids, str):
            pids = [x.strip() for x in pids.split(",") if x.strip()]
        if not isinstance(pids, list):
            errors.append(f"permission_ids must be a list for assignment {assignment.get('assignment_id')}")
            continue
        for pid in pids:
            if permission_ids and pid not in permission_ids:
                errors.append(f"Assignment references unknown permission {pid}")

    for event in events:
        if event.get("identity_id") not in identity_ids:
            errors.append(f"Event references unknown identity {event.get('identity_id')}")
        if not event.get("event_id"):
            errors.append("Event records must have a non-empty event_id value.")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str):
            errors.append(f"Event {event.get('event_id')} must have an ISO-8601 timestamp string.")
        else:
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"Event {event.get('event_id')} has an invalid timestamp.")
        if not isinstance(event.get("event_type"), str) or not event.get("event_type", "").strip():
            errors.append(f"Event {event.get('event_id')} must have a non-empty event_type.")
    return errors


def persist_bundle(root: Path, bundle: dict, *, backup: bool = True) -> None:
    """Persist a validated bundle, backing up existing files during explicit replacement."""
    errors = validate_bundle(bundle)
    if errors:
        raise ValueError("Bundle failed validation: " + "; ".join(errors[:8]))
    proc = root / "data" / "processed"
    raw = root / "data" / "raw" / "synthetic"
    proc.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    write_json(proc / "identities.json", bundle["identities"], backup=backup)
    write_json(proc / "assignments.json", bundle["assignments"], backup=backup)
    write_json(proc / "roles.json", bundle["roles"], backup=backup)
    write_json(proc / "permissions.json", bundle["permissions"], backup=backup)
    from .storage import atomic_write_text
    event_text = "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in bundle["events"])
    atomic_write_text(proc / "activity_events.jsonl", event_text, backup=backup)
    write_json(raw / "manifest.json", bundle["manifest"], backup=backup)


def ensure_project_data(root: Path | None = None, n: int = 1500, seed: int = 20261026) -> dict:
    root = root or ROOT
    proc = root / "data" / "processed"
    raw = root / "data" / "raw" / "synthetic"
    required = [proc / "identities.json", proc / "assignments.json", proc / "roles.json", proc / "permissions.json", proc / "activity_events.jsonl", raw / "manifest.json"]
    present = [path.exists() for path in required]
    if all(present):
        bundle = {"identities": read_json(proc / "identities.json", []), "assignments": read_json(proc / "assignments.json", []),
                  "roles": read_json(proc / "roles.json", []), "permissions": read_json(proc / "permissions.json", []),
                  "events": read_jsonl(proc / "activity_events.jsonl"), "manifest": read_json(raw / "manifest.json", {})}
        errors = validate_bundle(bundle)
        if errors:
            raise ValueError("Existing local data failed validation; it was not overwritten. Back up and repair the files, or explicitly generate a replacement dataset. Details: " + "; ".join(errors[:8]))
        return bundle
    if any(present):
        missing = [str(path.relative_to(root)) for path, exists in zip(required, present) if not exists]
        raise FileNotFoundError("DigitalGhost data is incomplete; existing files were preserved. Missing: " + ", ".join(missing))
    bundle = generate_synthetic_bundle(n=n, seed=seed)
    persist_bundle(root, bundle, backup=False)
    return bundle

def bundle_to_frame(bundle: dict) -> pd.DataFrame:
    """Create one feature row per identity while preserving multiple assignments."""
    identities = pd.DataFrame(bundle.get("identities", []))
    assignments = pd.DataFrame(bundle.get("assignments", []))
    if identities.empty:
        return identities
    if assignments.empty:
        identities["role_name"] = "Unassigned"
        identities["role_privilege"] = "Low"
        identities["role_id"] = "role:unassigned"
        identities["permission_ids"] = [[] for _ in range(len(identities))]
        identities["permission_names"] = [[] for _ in range(len(identities))]
        identities["permission_levels"] = [[] for _ in range(len(identities))]
        identities["resource_names"] = [[] for _ in range(len(identities))]
        identities["permission_used_last_30d"] = False
        identities["unused_high_privilege"] = False
    else:
        privilege_rank = {"low": 0, "medium": 1, "high": 2}
        def unique_join(values):
            output = []
            for value in values:
                if isinstance(value, str):
                    candidates = [value]
                elif isinstance(value, (list, tuple, set)):
                    candidates = list(value)
                else:
                    candidates = []
                for candidate in candidates:
                    text = str(candidate).strip()
                    if text and text not in output:
                        output.append(text)
            return output
        grouped = []
        for identity_id, group in assignments.groupby("identity_id", sort=False, dropna=False):
            levels = unique_join(group.get("permission_levels", pd.Series(dtype=object)).tolist())
            privilege = max(levels, key=lambda level: privilege_rank.get(level.lower(), -1), default="Low")
            high_assignments_unused = False
            for _, assignment in group.iterrows():
                assignment_levels = assignment.get("permission_levels", [])
                if isinstance(assignment_levels, str):
                    assignment_levels = [x.strip() for x in assignment_levels.split(",") if x.strip()]
                elif not isinstance(assignment_levels, (list, tuple, set)):
                    assignment_levels = []
                has_high = any(str(level).lower() == "high" for level in assignment_levels)
                used = assignment.get("permission_used_last_30d", False)
                used = str(used).strip().lower() in {"true", "1", "yes", "y"} if not isinstance(used, bool) else used
                if has_high and not used:
                    high_assignments_unused = True
            grouped.append({
                "identity_id": identity_id,
                "role_id": ", ".join(unique_join(group.get("role_id", pd.Series(dtype=object)).tolist())) or "role:unassigned",
                "role_name": ", ".join(unique_join(group.get("role_name", pd.Series(dtype=object)).tolist())) or "Unassigned",
                "role_privilege": privilege,
                "permission_ids": unique_join(group.get("permission_ids", pd.Series(dtype=object)).tolist()),
                "permission_names": unique_join(group.get("permission_names", pd.Series(dtype=object)).tolist()),
                "permission_levels": levels,
                "resource_names": unique_join(group.get("resource_names", pd.Series(dtype=object)).tolist()),
                "permission_used_last_30d": bool(group.get("permission_used_last_30d", pd.Series([False] * len(group))).map(lambda v: v if isinstance(v, bool) else str(v).strip().lower() in {"true", "1", "yes", "y"}).all()),
                "unused_high_privilege": high_assignments_unused,
            })
        aggregated = pd.DataFrame(grouped)
        identities = identities.merge(aggregated, on="identity_id", how="left", validate="one_to_one")
        for column in ["permission_ids", "permission_names", "permission_levels", "resource_names"]:
            if column in identities:
                identities[column] = identities[column].apply(lambda value: value if isinstance(value, list) else [])
        identities["role_name"] = identities.get("role_name", pd.Series(index=identities.index, dtype=object)).fillna("Unassigned")
        identities["role_privilege"] = identities.get("role_privilege", pd.Series(index=identities.index, dtype=object)).fillna("Low")
        identities["permission_used_last_30d"] = identities.get("permission_used_last_30d", pd.Series(False, index=identities.index)).fillna(False).astype(bool)
        identities["unused_high_privilege"] = identities.get("unused_high_privilege", pd.Series(False, index=identities.index)).fillna(False).astype(bool)
    identities["privilege_level"] = identities.get("permission_levels", pd.Series([[] for _ in range(len(identities))], index=identities.index)).apply(
        lambda values: max(values, key=lambda level: {"low": 0, "medium": 1, "high": 2}.get(str(level).lower(), -1), default="Low") if isinstance(values, list) else "Low")
    identities["role"] = identities.get("role_name", "Unassigned")
    identities["permission"] = identities.get("permission_names", pd.Series([[] for _ in range(len(identities))], index=identities.index)).apply(lambda values: ", ".join(values) if isinstance(values, list) else str(values))
    identities["permission_count"] = identities.get("permission_ids", pd.Series([[] for _ in range(len(identities))], index=identities.index)).apply(lambda values: len(values) if isinstance(values, list) else 0)
    return identities

def _audit_hash(record: dict) -> str:
    payload = {key: record[key] for key in ("timestamp", "action", "details", "previous_hash")}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_audit_hash(record: dict) -> str:
    payload = f"{record['timestamp']}|{record['action']}|{record['details']}|{record['previous_hash']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_audit(root: Path, action: str, details: dict) -> None:
    from .storage import append_jsonl, read_jsonl
    path = root / "data" / "state" / "audit.jsonl"
    previous = read_jsonl(path)
    integrity = verify_audit_chain(root)
    if not integrity["valid"]:
        raise ValueError(f"Refusing to append to an invalid audit chain (record {integrity['broken_at']}). Preserve the log and investigate before continuing.")
    prev_hash = previous[-1].get("record_hash", "GENESIS") if previous else "GENESIS"
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "action": str(action),
              "details": details, "previous_hash": prev_hash}
    record["record_hash"] = _audit_hash(record)
    append_jsonl(path, record)


def verify_audit_chain(root: Path) -> dict:
    """Verify record hashes and links; this detects edits but cannot prevent local tampering."""
    from .storage import read_jsonl
    records = read_jsonl(root / "data" / "state" / "audit.jsonl")
    previous_hash = "GENESIS"
    for index, record in enumerate(records, start=1):
        if record.get("previous_hash") != previous_hash:
            return {"valid": False, "records_checked": index - 1, "broken_at": index, "reason": "Previous-hash link does not match."}
        try:
            expected = _audit_hash(record)
            legacy_expected = _legacy_audit_hash(record)
        except (KeyError, TypeError, ValueError):
            return {"valid": False, "records_checked": index - 1, "broken_at": index, "reason": "Audit record is missing required fields."}
        # Accept records from v2.0's legacy hash format so existing local audit logs remain readable.
        if record.get("record_hash") not in {expected, legacy_expected}:
            return {"valid": False, "records_checked": index - 1, "broken_at": index, "reason": "Record hash does not match its contents."}
        previous_hash = record["record_hash"]
    return {"valid": True, "records_checked": len(records), "broken_at": None, "reason": "Audit chain is internally consistent."}
