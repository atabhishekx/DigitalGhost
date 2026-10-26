# DigitalGhost architecture notes

```text
Streamlit UI
  ├── Overview / Risk Explorer / Identity Graph
  ├── AI Investigator / Data & Models / Audit History
  └── consistent read-only review interface
             ↓
Analysis orchestration (analysis.py)
  ├── transparent rules (risk.py)
  ├── Isolation Forest + feature pipeline (ml.py)
  ├── identity-role-permission-resource graph (graph_analysis.py)
  └── transparent signal fusion (fusion.py)
             ↓
Read-only investigation orchestrator (agent.py)
  ├── identity context tool (tools.py)
  ├── rule, graph and behavior tools (tools.py)
  ├── deterministic evidence-backed report
  └── optional Gemini summary, disabled by default
             ↓
Local file storage (storage.py / data.py)
  ├── JSON / JSONL / CSV source files
  ├── atomic replacement, backups and validation
  ├── saved model artifacts and metadata
  └── append-oriented hash-linked audit history
```

## Core invariants

1. `identity_id`, `role_id`, `permission_id`, `assignment_id`, and `event_id` are stable identifiers; names are display labels only.
2. Multiple assignments for an identity are aggregated for ML features without silently dropping permissions.
3. Public imports and generated synthetic data remain separate until a deliberate, documented mapping is provided.
4. Source CSV bytes are preserved exactly and verified by SHA-256 in the manifest.
5. Synthetic scenario labels are excluded from model features and used only for held-out evaluation.
6. Rule, graph and anomaly signals remain separately visible; the combined priority is a triage heuristic, not a probability.
7. The investigation agent invokes bounded, read-only tools and has no permission-changing tools.
8. Every investigation, model training action, dataset import, synthetic evaluation and analyst decision is audited.
9. Local file persistence is intended for one user/process, not concurrent multi-user production.
