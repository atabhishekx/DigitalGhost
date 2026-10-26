"""Identity graph construction and explainable graph risk signals."""
from __future__ import annotations
import networkx as nx
import pandas as pd

SENSITIVE_WORDS = ("delete", "drop", "decrypt", "createuser", "passrole", "admin", "sensitive", "terminate", "writeall")

def build_identity_graph(identities: list[dict], assignments: list[dict], permissions: list[dict] | None = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    id_map = {x["identity_id"]: x for x in identities if x.get("identity_id")}
    permission_map = {x["permission_id"]: x for x in (permissions or []) if x.get("permission_id")}
    for ident in id_map.values():
        node_id = f"identity::{ident['identity_id']}"
        graph.add_node(node_id, node_type="identity", identity_id=ident["identity_id"], label=str(ident.get("name", ident["identity_id"])), lifecycle_status=ident.get("lifecycle_status", "Unknown"))
    for assignment in assignments:
        identity_id = assignment.get("identity_id")
        if identity_id not in id_map: continue
        identity_node = f"identity::{identity_id}"
        role_id = assignment.get("role_id") or "role::unassigned"
        role_node = f"role::{role_id}"
        graph.add_node(role_node, node_type="role", role_id=role_id, label=str(assignment.get("role_name", role_id)))
        graph.add_edge(identity_node, role_node, relation="assigned_role")
        permission_ids = assignment.get("permission_ids", [])
        if isinstance(permission_ids, str): permission_ids = [x.strip() for x in permission_ids.split(",") if x.strip()]
        for permission_id in permission_ids:
            permission_node = f"permission::{permission_id}"
            meta = permission_map.get(permission_id, {})
            label = meta.get("name") or permission_id
            graph.add_node(permission_node, node_type="permission", permission_id=permission_id, label=label,
                           privilege_level=meta.get("privilege_level", "Unknown"), resource_name=meta.get("resource_name", "Unknown"))
            graph.add_edge(role_node, permission_node, relation="grants_permission")
            # Assignments in this prototype represent effective permissions through a role;
            # do not mislabel inherited access as a direct assignment.
            resource_name = str(meta.get("resource_name", "Unknown")).strip()
            if resource_name and resource_name.lower() != "unknown":
                resource_node = f"resource::{resource_name.casefold()}"
                graph.add_node(resource_node, node_type="resource", label=resource_name, resource_name=resource_name)
                graph.add_edge(permission_node, resource_node, relation="targets_resource")
    return graph

def calculate_graph_risk(identities: list[dict], assignments: list[dict], permissions: list[dict] | None = None) -> pd.DataFrame:
    graph = build_identity_graph(identities, assignments, permissions)
    permission_users: dict[str, set[str]] = {}
    assignment_by_identity: dict[str, list[dict]] = {}
    for a in assignments:
        iid = a.get("identity_id")
        assignment_by_identity.setdefault(iid, []).append(a)
        pids = a.get("permission_ids", [])
        if isinstance(pids, str): pids = [x.strip() for x in pids.split(",") if x.strip()]
        for pid in pids: permission_users.setdefault(pid, set()).add(iid)
    pmeta = {p.get("permission_id"): p for p in (permissions or [])}
    output = []
    for ident in identities:
        iid = ident.get("identity_id")
        assigns = assignment_by_identity.get(iid, [])
        pids = [p for a in assigns for p in (a.get("permission_ids", []) if isinstance(a.get("permission_ids", []), list) else str(a.get("permission_ids", "")).split(","))]
        pids = [str(p).strip() for p in pids if str(p).strip()]
        high_count = 0; sensitive_count = 0; max_blast = 0; evidence = []
        sensitive_resources: set[str] = set()
        for pid in pids:
            p = pmeta.get(pid, {})
            level = str(p.get("privilege_level", "Unknown")).lower()
            name = str(p.get("name", pid)).lower()
            is_sensitive = level == "high" or any(word in name.replace(":", "") for word in SENSITIVE_WORDS)
            if level == "high": high_count += 1
            if is_sensitive:
                sensitive_count += 1
                resource_name = str(p.get("resource_name", "Unknown")).strip()
                if resource_name and resource_name.lower() != "unknown":
                    sensitive_resources.add(resource_name)
                users = len(permission_users.get(pid, set()))
                max_blast = max(max_blast, users)
                if users >= max(10, int(len(identities) * .08)):
                    evidence.append(f"Permission {p.get('name', pid)} is shared by {users} identities (broad blast radius).")
        broad_blast = max_blast >= max(10, int(len(identities) * .08))
        multi_resource_exposure = len(sensitive_resources) >= 3
        score = min(100, high_count * 12 + sensitive_count * 10 + (15 if broad_blast else 0) + (8 if multi_resource_exposure else 0))
        if multi_resource_exposure:
            evidence.append(f"Identity has sensitive access spanning {len(sensitive_resources)} distinct resources.")
        if str(ident.get("lifecycle_status", "")).lower() in {"terminated", "suspended"} and pids:
            score = min(100, score + 25)
            evidence.append("Non-active identity retains one or more permission assignments.")
        if high_count: evidence.append(f"Identity is assigned {high_count} high-privilege permission(s).")
        if not evidence: evidence.append("No graph-risk heuristic triggered for this identity.")
        output.append({"identity_id": iid, "graph_risk_score": score, "graph_high_privilege_count": high_count,
                       "graph_sensitive_permission_count": sensitive_count, "graph_sensitive_resource_count": len(sensitive_resources),
                       "graph_max_permission_blast_radius": max_blast, "graph_broad_blast_radius": bool(broad_blast),
                       "graph_reasons": evidence})
    return pd.DataFrame(output)
