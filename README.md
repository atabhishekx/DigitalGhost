# DigitalGhost — Local Identity Risk Intelligence

**DigitalGhost** is a local-first prototype for investigating potentially risky identity access and retained-access risk. It combines transparent rule-based scoring, a persisted Isolation Forest anomaly signal, graph-based permission analysis, a bounded read-only investigation agent, and an optional LLM summary — all running on your machine with no external identity provider and no automatic remediation.

![Identity lifecycle and retained-access risk](assets/identity-lifecycle.png)
*Identity lifecycle: old entitlements can persist after operational context changes.*

## What DigitalGhost does

DigitalGhost is built for security analysts and researchers who need a transparent, explainable view of identity risk — not a black-box score. Every signal is separately visible and explainable:

- **Rule engine** — transparent, configurable rules for terminated/suspended identities, inactivity, and unused high-privilege access. Each triggered rule lists its reason and points.
- **Behavioral ML** — a local scikit-learn Isolation Forest trained on identity activity features. Its percentile is a relative anomaly signal, not a probability.
- **Permission graph** — graph analytics over identities, roles, permissions, and resources to surface high-privilege assignments, blast radius, and multi-resource exposure.
- **Read-only AI Investigator** — a tool-using investigation workflow that gathers identity-context, rule, graph, and behavior evidence, saves evidence-backed reports, and can optionally draft a Gemini summary (disabled by default).
- **Audit history** — append-oriented, hash-linked log of investigations, model changes, dataset imports, and analyst decisions with an integrity check.

## Screenshots

### Identity lifecycle

![Identity lifecycle](assets/identity-lifecycle.png)

### Implemented system architecture

![System architecture](assets/system-architecture.png)

### Identity access graph

![Permission graph](assets/identity-access-graph.png)

### Synthetic benchmark pipeline

![Benchmark pipeline](assets/synthetic-benchmark.png)

### Traditional vs contextual analysis

![Traditional vs contextual](assets/traditional-vs-contextual.png)

## Key features

- Multi-page Streamlit UI with a consistent dark teal/blue theme across Overview, Risk Explorer, Identity Graph, AI Investigator, Data & Models, and Audit History workspaces.
- Stable typed identity, role, permission, assignment, event, and resource IDs — duplicate display names never merge graph nodes.
- Multiple assignments per identity aggregated for ML features without silently dropping data.
- Persisted Isolation Forest model with atomic artifact replacement, feature schema metadata, and artifact checksum. Scenario labels are excluded from model features.
- Public research CSV import with column detection, external labels preserved as annotations, and a 50 MB / 250,000-row limit.
- Held-out synthetic benchmark comparing Isolation Forest, rule-only, graph-only, and combined triage.
- Automated tests covering rules, data validation, graph IDs, ML leakage, model recovery, file persistence, provenance, audit integrity, CSV import, and investigation workflow.

## Quick start

```powershell
cd path\to\digitalghost
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The app opens at `http://localhost:8501`.

If PowerShell blocks environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Requirements

Python 3.10–3.12 recommended, a modern browser, and the packages in `requirements.txt`. No SQL server, external database, or API key is required for the core application. The first run creates local data and trains a local anomaly model.

Optional Gemini summaries require a `GEMINI_API_KEY`. The key is never written to project data files — do not commit secrets.

## Local file persistence

Application-owned state lives under the project directory:

- `data/processed/` — identities, assignments, roles, permissions, activity events
- `data/raw/synthetic/` — synthetic dataset manifest
- `data/raw/public/` — imported public research CSVs (kept separate from synthetic identities)
- `data/state/` — investigations, analyst decisions, evaluation reports, hash-linked audit history
- `models/` — saved Isolation Forest artifact and metadata

Generated data and local model artifacts are ignored by Git but remain on disk. Back up `data/` and `models/` if you want to preserve them across project copies.

## Safety boundaries

The investigator is **read-only**. It can gather local evidence and draft recommendations, but it cannot revoke access or change permissions. LLM summaries can be wrong and must be checked against the structured evidence. This prototype does not connect to a live identity provider.

## Project structure

```
digitalghost/
├── app.py                    # Streamlit UI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .env                      # Local environment (not committed)
├── .gitignore                # Git ignore rules
├── assets/                   # Images for README
│   ├── identity-lifecycle.png
│   ├── system-architecture.png
│   ├── identity-access-graph.png
│   ├── synthetic-benchmark.png
│   └── traditional-vs-contextual.png
├── data/                     # Local runtime data (not committed)
│   ├── processed/
│   ├── raw/
│   └── state/
├── models/                   # Trained ML artifacts (not committed)
├── src/digitalghost/         # Core Python package
│   ├── __init__.py
│   ├── agent.py              # Read-only investigation orchestrator
│   ├── analysis.py           # Signal orchestration
│   ├── data.py               # Data persistence and validation
│   ├── evaluation.py         # Synthetic benchmark evaluation
│   ├── fusion.py             # Signal fusion logic
│   ├── graph_analysis.py     # Permission graph analytics
│   ├── ml.py                 # Isolation Forest model
│   ├── public_import.py      # Public CSV import
│   ├── risk.py               # Rule engine
│   ├── storage.py            # File-backed storage
│   └── tools.py              # Investigation tools
├── tests/                    # Pytest test suite
└── docs/
    └── ARCHITECTURE.md       # Architecture notes
```

## License

This is a college project prototype for educational and research purposes. It is not a production security tool.

## Git setup (new repository)

```powershell
git init
git add .
git commit -m "Initial commit: DigitalGhost local identity risk intelligence"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/digitalghost.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and create the empty `digitalghost` repository on GitHub first.