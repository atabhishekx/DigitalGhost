from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from digitalghost.data import ensure_project_data, bundle_to_frame, validate_bundle, write_audit, generate_synthetic_bundle, persist_bundle, verify_audit_chain
from digitalghost.analysis import analyze_all
from digitalghost.graph_analysis import build_identity_graph
from digitalghost.ml import train_model, MODEL_PATH, META_PATH
from digitalghost.agent import investigate, record_analyst_decision
from digitalghost.public_import import normalize_public_csv, save_public_import
from digitalghost.evaluation import evaluate_synthetic_bundle, save_evaluation_report
from digitalghost.storage import read_json, read_jsonl, write_json

st.set_page_config(page_title="DigitalGhost | Identity Risk Intelligence", page_icon="👻", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --dg-accent: #58d6c5; --dg-ink: #edf2f7; --dg-muted: #9aa9b9; --dg-panel: #111d30; --dg-border: #26364b; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; }
.stApp { background: #0b1220; color: var(--dg-ink); }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid var(--dg-border); }
[data-testid="stSidebar"] * { font-size: 0.90rem; }
h1, h2, h3, h4 { letter-spacing: -0.025em; line-height: 1.28; }
h1 { font-size: 1.78rem !important; font-weight: 700 !important; margin-bottom: .25rem !important; }
h2 { font-size: 1.34rem !important; font-weight: 650 !important; }
h3 { font-size: 1.05rem !important; font-weight: 650 !important; }
p, li, label, .stMarkdown, [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] { font-size: 0.90rem; line-height: 1.55; }
small, [data-testid="stWidgetLabel"] p { font-size: 0.88rem !important; }
[data-testid="stMetric"] { background: var(--dg-panel); border: 1px solid var(--dg-border); padding: 14px 16px; border-radius: 12px; }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #aab8c8 !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { border-radius: 9px; border: 1px solid #315064; min-height: 2.45rem; font-size: .90rem; font-weight: 600; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background: var(--dg-accent); color: #07151a; border: 0; }
[data-testid="stDataFrame"] { border: 1px solid var(--dg-border); border-radius: 10px; overflow: hidden; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div, [data-testid="stMultiSelect"] div { font-size: .90rem; }
[data-testid="stExpander"] { border: 1px solid var(--dg-border); border-radius: 10px; }
.dg-kicker { color: var(--dg-accent); font-size: .74rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .25rem; }
.dg-muted { color: var(--dg-muted); font-size: .92rem; line-height: 1.55; }
.dg-card { background: var(--dg-panel); border: 1px solid var(--dg-border); padding: 16px 18px; border-radius: 12px; }
hr { border-color: var(--dg-border); }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading local project data…")
def load_bundle():
    return ensure_project_data(ROOT)

@st.cache_data(show_spinner="Analyzing identities, graph relationships and anomaly signals…")
def run_analysis(bundle, wt, wi, wh, threshold, model_mtime):
    return analyze_all(bundle, wt, wi, wh, threshold, model_path=MODEL_PATH)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Local data could not be loaded safely: {exc}")
    st.info("Check the data/processed files. A valid backup may be restored from the matching .bak file.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="dg-kicker">Local-first security analytics</div>', unsafe_allow_html=True)
    st.title("👻 DigitalGhost")
    st.caption("Identity Risk Intelligence · v2.1")
    page = st.radio("WORKSPACE", ["Overview", "Risk Explorer", "Identity Graph", "AI Investigator", "Data & Models", "Audit History"], label_visibility="visible")
    st.divider()
    st.markdown("**Rule settings**")
    wt = st.slider("Terminated / suspended identity", 0, 100, 45, 5)
    wi = st.slider("Dormant identity", 0, 100, 30, 5)
    wh = st.slider("Unused high privilege", 0, 100, 25, 5)
    threshold = st.slider("Inactivity threshold (days)", 30, 365, 90, 5)
    st.caption("Rule weights are heuristic points, not incident probabilities.")

model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
try:
    analyzed = run_analysis(bundle, wt, wi, wh, threshold, model_mtime)
except Exception as exc:
    st.error(f"Analysis could not complete: {exc}")
    st.stop()
model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
if model_mtime_after and model_mtime_after != model_mtime:
    try:
        model_meta = read_json(META_PATH, {}) or {}
        write_audit(ROOT, "model_auto_trained_or_recovered", {"model_type": model_meta.get("model_type", "IsolationForest"), "trained_rows": model_meta.get("trained_rows", len(bundle.get("identities", []))), "artifact_sha256": model_meta.get("artifact_sha256")})
    except Exception as audit_exc:
        st.warning(f"Model is available, but its automatic training event could not be appended to the audit log ({type(audit_exc).__name__}). Review Audit History before making decisions.")

# Reusable header
def page_header(kicker: str, title: str, subtitle: str):
    st.markdown(f'<div class="dg-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="dg-muted">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")

def show_metrics(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Identities", f"{len(df):,}")
    c2.metric("Critical rule findings", f"{int((df.risk_category == 'Critical Risk').sum()):,}")
    c3.metric("ML anomalies", f"{int(df.is_anomaly.sum()):,}")
    c4.metric("High / urgent priority", f"{int(df.priority_band.isin(['High','Urgent']).sum()):,}")

def display_cols(df):
    cols = ["identity_id", "name", "department", "lifecycle_status", "role", "permission_count", "risk_score", "risk_category", "graph_risk_score", "anomaly_percentile", "priority_score", "priority_band"]
    return df[[c for c in cols if c in df.columns]]

if page == "Overview":
    page_header("COMMAND CENTER", "Identity risk overview", "A local-first view of retained access, behavioral anomalies and graph-derived risk signals.")
    st.warning("Findings are indicators for human review, not confirmed incidents. This prototype does not revoke access or connect to live identity systems.")
    show_metrics(analyzed)
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Highest-priority identities")
        st.dataframe(display_cols(analyzed.head(12)), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Priority distribution")
        counts = analyzed["priority_band"].value_counts().reindex(["Urgent", "High", "Review", "Routine"], fill_value=0)
        st.bar_chart(counts, horizontal=True, use_container_width=True)
        st.caption("Priority is a triage heuristic combining separate signals; it is not a calibrated probability.")
    st.subheader("Signal interpretation")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="dg-card"><b>Rule engine · What triggered?</b><p>Checks lifecycle status, inactivity and unused high-privilege permissions. Each triggered rule lists the reason and adds configured points.</p></div>', unsafe_allow_html=True)
    s2.markdown('<div class="dg-card"><b>Behavioral ML · What looks unusual?</b><p>Isolation Forest compares feature patterns. Its percentile is a relative anomaly signal, not a probability, and does not by itself explain the cause.</p></div>', unsafe_allow_html=True)
    s3.markdown('<div class="dg-card"><b>Permission graph · What access stands out?</b><p>Checks high-privilege access, sensitive permissions, broad sharing and retained assignments. Findings show the specific heuristic that triggered.</p></div>', unsafe_allow_html=True)
    with st.expander("How to interpret the dashboard scores", expanded=False):
        st.markdown("""- **Rule score:** points from explicit, configurable rules; capped at 100.
- **Anomaly percentile:** how unusual a feature pattern is relative to scored records; not a compromise probability.
- **Graph score:** points from structural permission heuristics; not proof of an exploitable path.
- **Triage priority:** a combined review-order heuristic. Always inspect the underlying evidence before deciding.""")

elif page == "Risk Explorer":
    page_header("INVESTIGATE", "Risk explorer", "Filter, sort and inspect individual identity findings with separate, explainable signals.")
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1: categories = st.multiselect("Rule category", ["Critical Risk", "Moderate Risk", "Low Risk", "Minimal Risk"], default=["Critical Risk", "Moderate Risk", "Low Risk", "Minimal Risk"])
    with c2: priorities = st.multiselect("Priority band", ["Urgent", "High", "Review", "Routine"], default=["Urgent", "High", "Review"])
    with c3: query = st.text_input("Search identity / role / permission", placeholder="e.g. id:000100 or Cloud Administrator")
    filtered = analyzed[analyzed.risk_category.isin(categories) & analyzed.priority_band.isin(priorities)].copy()
    if query.strip():
        q = query.strip().lower()
        searchable = filtered.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False, na=False)).any(axis=1)
        filtered = filtered[searchable]
    st.caption(f"Showing {len(filtered):,} of {len(analyzed):,} identities")
    st.dataframe(display_cols(filtered), use_container_width=True, hide_index=True)
    if not filtered.empty:
        selected_id = st.selectbox("Inspect identity", filtered.identity_id.tolist(), format_func=lambda x: f"{x} · {filtered.loc[filtered.identity_id == x, 'name'].iloc[0]}")
        row = filtered[filtered.identity_id == selected_id].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Rule score", f"{int(row.risk_score)}/100", str(row.risk_category))
        b.metric("Graph risk", f"{int(row.graph_risk_score)}/100")
        c.metric("Anomaly percentile", f"{row.anomaly_percentile:.1f}")
        st.subheader("Why this identity received these signals")
        st.caption("Each signal answers a different question. Read the evidence before deciding whether the account needs action.")
        rule_tab, ml_tab, graph_tab, priority_tab = st.tabs(["1 · Rule checks", "2 · Behavioral ML", "3 · Permission graph", "4 · Combined priority"])
        with rule_tab:
            st.markdown("**Question answered:** Does this identity match one of the explicit risk rules?")
            st.metric("Rule score", f"{int(row.risk_score)}/100", str(row.risk_category))
            rule_reasons = list(row.reasons_list or [])
            if rule_reasons:
                for reason in rule_reasons:
                    st.markdown(f"- {reason}")
            else:
                st.success("No configured rule triggered for this identity.")
            st.caption("The score adds configured points for lifecycle status, inactivity and unused high-privilege access, then caps at 100. It is not a probability.")
        with ml_tab:
            st.markdown("**Question answered:** Does the observed feature pattern look unusual compared with patterns learned by Isolation Forest?")
            ml_a, ml_b = st.columns(2)
            ml_a.metric("Anomaly percentile", f"{float(row.anomaly_percentile):.1f}/100")
            ml_b.metric("Flagged at model threshold", "Yes" if bool(row.is_anomaly) else "No")
            if bool(row.is_anomaly):
                st.warning("The model marked this record as unusual relative to its learned distribution. This is a signal for review, not proof of malicious activity.")
            else:
                st.success("The model did not flag this record at its current threshold. This does not guarantee the identity is safe.")
            st.markdown("**What this score means**")
            st.write("A higher percentile means the record is more unusual relative to the scored dataset. Isolation Forest does not automatically explain which feature caused the result, so DigitalGhost does not claim a specific feature caused this anomaly.")
            feature_cols = [c for c in ["last_activity_days", "failed_logins_7d", "unique_resources_7d", "privileged_actions_7d", "off_hours_events_7d", "permission_count"] if c in row.index]
            if feature_cols:
                st.markdown("**Input features considered**")
                st.dataframe(pd.DataFrame([{"Feature": c.replace("_", " ").title(), "Observed value": row[c], "How to read it": {"last_activity_days": "Days since recorded activity", "failed_logins_7d": "Failed logins in the last 7 days", "unique_resources_7d": "Distinct resources accessed in the last 7 days", "privileged_actions_7d": "Privileged actions in the last 7 days", "off_hours_events_7d": "Events outside typical working hours in the last 7 days", "permission_count": "Number of assigned permissions"}.get(c, "Model input feature")} for c in feature_cols]), use_container_width=True, hide_index=True)
        with graph_tab:
            st.markdown("**Question answered:** Do this identity's permissions and access relationships match graph-risk heuristics?")
            st.metric("Graph risk score", f"{int(row.graph_risk_score)}/100")
            graph_reasons = list(row.graph_reasons or [])
            if graph_reasons:
                for reason in graph_reasons:
                    st.markdown(f"- {reason}")
            st.markdown("**Graph facts for this identity**")
            g1, g2, g3 = st.columns(3)
            g1.metric("High-privilege permissions", int(getattr(row, "graph_high_privilege_count", 0)))
            g2.metric("Sensitive permissions", int(getattr(row, "graph_sensitive_permission_count", 0)))
            g3.metric("Sensitive resources", int(getattr(row, "graph_sensitive_resource_count", 0)))
            st.caption("Graph risk is based on configured heuristics, such as high-privilege permissions, broad sharing and retained assignments for non-active identities. It is not a confirmed attack path.")
        with priority_tab:
            st.markdown("**Question answered:** Which identities should be reviewed first when all available signals are considered?")
            st.metric("Triage priority score", f"{float(row.priority_score):.1f}/100", str(row.priority_band))
            st.write("This is a ranking aid combining rule, graph and anomaly signals. It is not a calibrated likelihood of compromise.")
            st.markdown("**Signal breakdown**")
            st.dataframe(pd.DataFrame([
                {"Signal": "Rule engine", "Value": f"{int(row.risk_score)}/100", "Interpretation": str(row.risk_category)},
                {"Signal": "Permission graph", "Value": f"{int(row.graph_risk_score)}/100", "Interpretation": f"{int(getattr(row, 'graph_sensitive_permission_count', 0))} sensitive permission(s)"},
                {"Signal": "Behavioral ML", "Value": f"{float(row.anomaly_percentile):.1f} percentile", "Interpretation": "Flagged as unusual" if bool(row.is_anomaly) else "Not flagged at current threshold"},
            ]), use_container_width=True, hide_index=True)
            st.markdown("**Recommended human checks**")
            st.markdown("1. Verify the identity's lifecycle status.  2. Confirm current role and permission assignments.  3. Check relevant activity logs.  4. Record the analyst's decision before any access change.")
        with st.expander("Show raw evidence record (for technical review)", expanded=False):
            st.json({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in row.to_dict().items()}, expanded=False)
        st.download_button("Download identity evidence (JSON)", json.dumps({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in row.to_dict().items()}, indent=2, default=str), file_name=f"{selected_id.replace(':','_')}_evidence.json", mime="application/json")
    else: st.info("No identities match these filters.")

elif page == "Identity Graph":
    page_header("RELATIONSHIPS", "Identity access graph", "Explore unique identity IDs, roles and permissions without merging people who share a display name.")
    graph = build_identity_graph(bundle["identities"], bundle["assignments"], bundle["permissions"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", f"{graph.number_of_nodes():,}"); c2.metric("Relationships", f"{graph.number_of_edges():,}"); c3.metric("Connected components", f"{nx.number_weakly_connected_components(graph):,}" if graph.number_of_nodes() else "0")
    max_nodes = st.slider("Maximum graph nodes to display", 50, min(500, max(50, graph.number_of_nodes())), min(180, max(50, graph.number_of_nodes())), 10)
    # Keep graph rendering responsive: choose the highest-degree nodes, then include their neighbors.
    ranked = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)[:max_nodes]
    view = graph.subgraph(ranked).copy()
    if view.number_of_nodes() > 0:
        pos = nx.spring_layout(view, seed=42, k=0.8 / max(1, len(view.nodes) ** .25))
        fig, ax = plt.subplots(figsize=(13, 7), facecolor="#0b1220")
        ax.set_facecolor("#0b1220")
        colors = []
        for node in view.nodes:
            t = view.nodes[node].get("node_type")
            colors.append({"identity": "#58d6c5", "role": "#7487a5", "permission": "#e7b86b"}.get(t, "#9aa9b9"))
        nx.draw_networkx_nodes(view, pos, node_color=colors, node_size=270, alpha=.95, ax=ax)
        nx.draw_networkx_edges(view, pos, edge_color="#526178", arrows=False, alpha=.42, width=.7, ax=ax)
        labels = {n: str(view.nodes[n].get("label", n))[:22] for n in view.nodes}
        nx.draw_networkx_labels(view, pos, labels=labels, font_size=6, font_color="#eaf0f7", ax=ax)
        ax.axis("off"); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption("Teal = identity · Slate = role · Gold = permission/resource. The visualization is sampled by degree for responsiveness; graph-risk analysis uses the full local graph.")
    st.dataframe(pd.DataFrame([{"node_id": n, **attrs} for n, attrs in list(graph.nodes(data=True))[:100]]), use_container_width=True, hide_index=True)

elif page == "AI Investigator":
    page_header("INVESTIGATION WORKSPACE", "AI investigator", "Run a bounded, read-only investigation using rule, anomaly and graph evidence. LLM support is optional.")
    st.info("The investigator cannot revoke access or mutate identity permissions. LLM output is treated as a draft summary and must be checked against evidence.")
    selected_id = st.selectbox("Identity to investigate", analyzed.identity_id.tolist(), format_func=lambda x: f"{x} · {analyzed.loc[analyzed.identity_id == x, 'name'].iloc[0]}")
    use_llm = st.checkbox("Enhance report with Gemini (requires GEMINI_API_KEY)", value=False)
    api_key = st.text_input("Gemini API key (optional; not saved to project files)", type="password") if use_llm else None
    if st.button("Run investigation", type="primary"):
        try:
            report = investigate(selected_id, analyzed, bundle=bundle, use_llm=use_llm, api_key=api_key or None, root=ROOT)
            st.session_state["last_investigation"] = report
            st.success("Investigation saved locally.")
        except Exception as exc: st.error(f"Investigation failed safely: {exc}")
    report = st.session_state.get("last_investigation")
    if report:
        st.subheader(f"Report · {report['investigation_id']}")
        st.markdown(f"**Assessment:** {report['assessment']}")
        st.markdown("**Identity context**")
        st.json(report.get("identity_context", {}), expanded=False)
        st.markdown("**Evidence**")
        for item in report["evidence"]: st.markdown(f"- {item}")
        with st.expander("Investigation tool trace", expanded=False):
            st.dataframe(pd.DataFrame(report.get("tool_trace", [])), use_container_width=True, hide_index=True)
        if report.get("llm_summary"): st.markdown("**LLM draft summary**"); st.write(report["llm_summary"])
        st.caption(report.get("llm_note", ""))
        st.markdown("**Recommended review steps**")
        for item in report["recommended_steps"]: st.markdown(f"- {item}")
        st.download_button("Download investigation report", json.dumps(report, indent=2, default=str), file_name=f"{report['investigation_id']}.json", mime="application/json")
        with st.form("decision_form"):
            decision = st.selectbox("Analyst decision", ["Needs review", "Confirmed risk", "False positive", "Resolved"])
            notes = st.text_area("Analyst notes", max_chars=2000)
            submitted = st.form_submit_button("Save analyst decision")
        if submitted:
            record_analyst_decision(report["investigation_id"], decision, notes, ROOT)
            st.success("Analyst decision saved to local state and audit history.")
    history = read_json(ROOT / "data/state/investigations.json", []) or []
    st.subheader("Saved investigations")
    if history:
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in ["investigation_id", "created_at", "identity_id", "identity_name", "rule_score", "graph_risk_score", "anomaly_percentile", "llm_enhanced"]} for r in history[-100:][::-1]]), use_container_width=True, hide_index=True)
    else: st.caption("No investigations have been saved yet.")

elif page == "Data & Models":
    page_header("LOCAL DATA CONTROL", "Data & model management", "All state is stored in project files. Public imports are isolated from generated demo identities.")
    identities = bundle.get("identities", []); events = bundle.get("events", [])
    c1, c2, c3 = st.columns(3); c1.metric("Synthetic identities", f"{len(identities):,}"); c2.metric("Synthetic events", f"{len(events):,}"); c3.metric("Public imports", f"{len(list((ROOT/'data/raw/public').glob('*/manifest.json'))):,}")
    with st.expander("Generate a new synthetic dataset", expanded=False):
        st.warning("This replaces the active synthetic identities, assignments and events. Existing files are backed up as .bak; public imports and investigation history are preserved. New identity IDs include the seed so old reports do not silently refer to new records.")
        with st.form("regenerate_synthetic_form"):
            new_identity_count = st.number_input("Number of identities", min_value=500, max_value=10000, value=max(500, min(10000, len(identities))), step=500)
            new_events_per_identity = st.number_input("Events per identity", min_value=1, max_value=20, value=8, step=1)
            new_seed = st.number_input("Generator seed", min_value=0, max_value=2147483647, value=20261026, step=1)
            confirm_replace = st.checkbox("I understand the active synthetic dataset will be replaced.")
            regenerate = st.form_submit_button("Generate & save dataset", type="primary", disabled=not confirm_replace)
        if regenerate:
            try:
                with st.spinner("Generating, validating and saving the new dataset…"):
                    replacement_bundle = generate_synthetic_bundle(int(new_identity_count), int(new_events_per_identity), int(new_seed))
                    # Clear cached source data before replacing files, so a failed replacement cannot leave stale data cached.
                    st.cache_data.clear()
                    persist_bundle(ROOT, replacement_bundle, backup=True)
                    # Force a clean model build for the new dataset; if training fails, the next analysis can retrain.
                    try:
                        metadata = train_model(bundle_to_frame(replacement_bundle))
                    except Exception:
                        MODEL_PATH.unlink(missing_ok=True)
                        META_PATH.unlink(missing_ok=True)
                        raise
                    write_audit(ROOT, "model_trained_after_dataset_regeneration", {"model_type": metadata["model_type"], "trained_rows": metadata["trained_rows"], "artifact_sha256": metadata.get("artifact_sha256")})
                    write_audit(ROOT, "synthetic_dataset_regenerated", {"dataset_id": replacement_bundle["manifest"]["dataset_id"], "identity_count": len(replacement_bundle["identities"]), "event_count": len(replacement_bundle["events"]), "seed": int(new_seed)})
                st.cache_data.clear()
                st.success(f"Saved {len(replacement_bundle['identities']):,} identities and {len(replacement_bundle['events']):,} events. Previous generated files were backed up.")
                st.rerun()
            except Exception as exc:
                st.error(f"Dataset regeneration failed: {exc}")
    st.subheader("Dataset provenance")
    st.json(bundle.get("manifest", {}), expanded=False)
    st.subheader("Train / refresh anomaly model")
    st.write("The model is trained locally and saved under `models/`. Scenario labels are excluded from model features. Retraining replaces the local model and writes metadata.")
    if st.button("Train Isolation Forest", type="primary"):
        try:
            with st.spinner("Training and saving local model…"):
                frame = bundle_to_frame(bundle)
                metadata = train_model(frame)
                st.cache_data.clear()
            write_audit(ROOT, "model_trained", {"model_type": metadata["model_type"], "trained_rows": metadata["trained_rows"], "feature_data_sha256": metadata["feature_data_sha256"]})
            st.success(f"Model trained on {metadata['trained_rows']:,} identity records and saved to models/isolation_forest.joblib.")
            st.json(metadata)
        except Exception as exc: st.error(f"Model training failed: {exc}")
    if META_PATH.exists():
        st.subheader("Current model metadata")
        st.json(read_json(META_PATH, {}), expanded=False)
    else: st.warning("No saved model metadata found. The first analysis run trains a model automatically.")

    st.subheader("Held-out synthetic benchmark")
    st.caption("Compares Isolation Forest, rule-only, graph-only and combined triage on a held-out identity split. The synthetic scenario labels are evaluation targets only; they are excluded from model features.")
    if st.button("Run synthetic benchmark", type="primary"):
        try:
            with st.spinner("Training a temporary model and evaluating held-out identities…"):
                evaluation_report = evaluate_synthetic_bundle(bundle)
                save_evaluation_report(ROOT, evaluation_report)
                st.session_state["last_evaluation_report"] = evaluation_report
            st.success("Benchmark completed and report saved locally.")
        except Exception as exc:
            st.error(f"Benchmark could not run: {exc}")
    evaluation_report = st.session_state.get("last_evaluation_report")
    if evaluation_report is None:
        saved_evaluations = read_json(ROOT / "data/state/evaluation_reports.json", []) or []
        evaluation_report = saved_evaluations[-1] if saved_evaluations else None
    if evaluation_report:
        st.caption(f"Test set: {evaluation_report['test_identities']:,} identities · Positive synthetic scenarios: {evaluation_report['positive_test_cases']:,}")
        st.dataframe(pd.DataFrame(evaluation_report["methods"]), use_container_width=True, hide_index=True)
        st.info(evaluation_report["limitations"])

    st.subheader("Import public research CSV")
    st.markdown("Suggested small benchmark: [CERT-derived labeled event CSV on Zenodo](https://zenodo.org/records/19135764). Download `cert_preprocessed_ml_comparison.csv`, review the dataset page/license, then upload it below. This is an insider-activity benchmark, not ground truth for excessive cloud permissions.")
    st.caption("Supports common activity CSVs with a recognizable user column (user, username, user_id, identity). Imported data is kept separate; it is not automatically merged with synthetic identities or treated as ground truth.")
    uploaded = st.file_uploader("Choose public CSV", type=["csv"], help="Maximum 50 MB and 250,000 rows. Check the dataset license before importing.")
    source_name = st.text_input("Dataset name", value="Public research dataset")
    source_url = st.text_input("Source URL (optional)")
    license_text = st.text_input("License / usage note", value="Verify source license before use")
    if st.button("Validate and import CSV", disabled=uploaded is None):
        try:
            content = uploaded.getvalue()
            normalized, manifest = normalize_public_csv(content, source_name, source_url, license_text)
            outdir = save_public_import(ROOT, content, normalized, manifest)
            st.success(f"Imported {len(normalized):,} normalized events to {outdir.relative_to(ROOT)}.")
            st.json(manifest, expanded=False)
            st.cache_data.clear()
        except Exception as exc: st.error(f"Import rejected: {exc}")
    st.subheader("Data integrity checks")
    errors = validate_bundle(bundle)
    if errors: st.error("\n".join(errors[:20]))
    else: st.success("Identity IDs are unique; assignments and events reference known identities; required lifecycle fields are valid.")
    if st.button("Export analyzed findings CSV"):
        st.download_button("Download CSV", analyzed.to_csv(index=False).encode("utf-8"), file_name="digitalghost_findings.csv", mime="text/csv")

elif page == "Audit History":
    page_header("TRACEABILITY", "Audit history", "A local append-oriented log of investigations, model changes, dataset imports and analyst decisions.")
    try:
        audit = read_jsonl(ROOT / "data/state/audit.jsonl")
        chain_status = verify_audit_chain(ROOT)
    except Exception as exc:
        audit = []
        chain_status = {"valid": False, "broken_at": "unknown", "reason": f"Audit log could not be parsed ({type(exc).__name__}). Preserve the file before attempting repair."}
    decisions = read_json(ROOT / "data/state/analyst_decisions.json", []) or []
    c1, c2 = st.columns(2); c1.metric("Audit records", f"{len(audit):,}"); c2.metric("Analyst decisions", f"{len(decisions):,}")
    if chain_status["valid"]:
        st.success(f"Audit integrity check passed for {chain_status['records_checked']:,} records. This detects accidental edits but is not tamper-proof against a person who controls the project files.")
    else:
        st.error(f"Audit integrity check failed at record {chain_status['broken_at']}: {chain_status['reason']}")
    if audit:
        st.dataframe(pd.DataFrame(audit[::-1]), use_container_width=True, hide_index=True)
        st.download_button("Export audit JSONL", "\n".join(json.dumps(x, ensure_ascii=False) for x in audit) + "\n", file_name="digitalghost_audit.jsonl", mime="application/x-ndjson")
    else: st.info("No audit records yet. Running an investigation, importing a dataset or training a model creates an audit entry.")
    if decisions:
        st.subheader("Analyst decisions")
        st.dataframe(pd.DataFrame(decisions[::-1]), use_container_width=True, hide_index=True)

st.divider()
st.caption("DigitalGhost v2.1 · Local file persistence · Human-reviewed security indicators · No automatic remediation")