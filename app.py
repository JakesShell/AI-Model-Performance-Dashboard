from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.recommendation_engine import build_executive_summary, build_shadow_summary, get_best_models
from services.risk_engine import build_risk_matrix, incident_priority_summary
from services.scoring_engine import calculate_model_scores

st.set_page_config(
    page_title="AI Model Reliability Command Center",
    page_icon="ðŸ§ ",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(f"{DATA_PATH}/model_registry.csv")
    metrics = pd.read_csv(f"{DATA_PATH}/model_metrics.csv")
    incidents = pd.read_csv(f"{DATA_PATH}/model_incidents.csv")
    audit = pd.read_csv(f"{DATA_PATH}/governance_audit.csv")
    shadow = pd.read_csv(f"{DATA_PATH}/shadow_deployment.csv")
    return registry, metrics, incidents, audit, shadow


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 18% 0%, rgba(67, 97, 238, 0.34), transparent 28%),
                    radial-gradient(circle at 85% 10%, rgba(124, 58, 237, 0.26), transparent 30%),
                    linear-gradient(135deg, #07111f 0%, #0b1220 45%, #050914 100%);
                color: #e5eefc;
            }

            section[data-testid="stSidebar"] {
                background: rgba(5, 12, 24, 0.84);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-card {
                padding: 28px;
                border-radius: 28px;
                border: 1px solid rgba(148, 163, 184, 0.22);
                background:
                    linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(17, 24, 39, 0.58)),
                    radial-gradient(circle at 82% 20%, rgba(56, 189, 248, 0.20), transparent 26%);
                box-shadow: 0 24px 80px rgba(2, 6, 23, 0.44);
            }

            .eyebrow {
                color: #93c5fd;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.76rem;
                font-weight: 800;
                margin-bottom: 0.6rem;
            }

            .hero-title {
                font-size: clamp(2.1rem, 4vw, 4.2rem);
                line-height: 1.02;
                margin: 0;
                color: #f8fbff;
                letter-spacing: -0.07em;
                font-weight: 850;
            }

            .hero-copy {
                color: #b9c7dc;
                max-width: 950px;
                margin-top: 16px;
                font-size: 1.02rem;
                line-height: 1.7;
            }

            .metric-card {
                padding: 20px;
                border-radius: 22px;
                border: 1px solid rgba(148, 163, 184, 0.20);
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.54));
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 18px 45px rgba(2, 6, 23, 0.28);
                min-height: 130px;
            }

            .metric-label {
                color: #94a3b8;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-weight: 700;
            }

            .metric-value {
                color: #f8fafc;
                font-size: 2.1rem;
                font-weight: 820;
                margin-top: 6px;
            }

            .metric-note {
                color: #93c5fd;
                font-size: 0.86rem;
                margin-top: 6px;
            }

            .panel {
                padding: 22px;
                border-radius: 24px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: rgba(15, 23, 42, 0.58);
                box-shadow: 0 18px 45px rgba(2, 6, 23, 0.22);
            }

            .status-pill {
                display: inline-block;
                padding: 7px 10px;
                border-radius: 999px;
                background: rgba(14, 165, 233, 0.14);
                border: 1px solid rgba(125, 211, 252, 0.28);
                color: #bae6fd;
                font-size: 0.8rem;
                font-weight: 700;
                margin-right: 8px;
                margin-bottom: 8px;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }

            h1, h2, h3 {
                color: #f8fafc;
                letter-spacing: -0.03em;
            }

            p, li, label, span {
                color: #cbd5e1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def filter_scores(scores: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("AI Governance Filters")
    st.sidebar.caption("Shape the command center view by team, model type, and deployment stage.")

    model_types = st.sidebar.multiselect(
        "Model Type",
        sorted(scores["model_type"].unique()),
        default=sorted(scores["model_type"].unique()),
    )
    stages = st.sidebar.multiselect(
        "Deployment Stage",
        sorted(scores["deployment_stage"].unique()),
        default=sorted(scores["deployment_stage"].unique()),
    )
    owners = st.sidebar.multiselect(
        "Owning Team",
        sorted(scores["owner"].unique()),
        default=sorted(scores["owner"].unique()),
    )
    only_high_risk = st.sidebar.toggle("Show High/Critical Risk Only", value=False)

    filtered = scores[
        scores["model_type"].isin(model_types)
        & scores["deployment_stage"].isin(stages)
        & scores["owner"].isin(owners)
    ].copy()

    if only_high_risk:
        filtered = filtered[filtered["risk_status"].isin(["High", "Critical"])]

    return filtered


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">ENTERPRISE AI RELIABILITY | GOVERNANCE | DEPLOYMENT READINESS</div>
            <h1 class="hero-title">AI Model Reliability And Governance Command Center</h1>
            <div class="hero-copy">
                A production-style AI operations dashboard for monitoring model quality, drift, business risk,
                LLM safety, cost efficiency, incidents, and approval readiness before models reach production.
            </div>
            <div style="margin-top: 18px;">
                <span class="status-pill">Model Registry</span>
                <span class="status-pill">Deployment Gates</span>
                <span class="status-pill">Drift Detection</span>
                <span class="status-pill">LLM Safety</span>
                <span class="status-pill">Governance Audit Trail</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_view(scores: pd.DataFrame, incidents: pd.DataFrame) -> None:
    st.subheader("Executive Command Center")
    winners = get_best_models(scores)

    total_models = len(scores)
    approved = scores[scores["recommended_action"].str.contains("Approved", case=False, na=False)].shape[0]
    high_risk = scores[scores["risk_status"].isin(["High", "Critical"])].shape[0]
    avg_health = scores["health_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Models Monitored", str(total_models), "Across ML and LLM systems")
    with c2:
        metric_card("Approved Models", str(approved), "Production or shadow-ready")
    with c3:
        metric_card("High-Risk Models", str(high_risk), "Needs governance attention")
    with c4:
        metric_card("Average Health", f"{avg_health:.1f}", "Portfolio reliability score")

    st.markdown("### Executive Summary")
    st.markdown(f"<div class='panel'>{build_executive_summary(scores)}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        fig = px.scatter(
            scores,
            x="deployment_readiness_score",
            y="risk_score",
            size="monthly_predictions",
            color="risk_status",
            hover_name="model_name",
            hover_data=["version", "owner", "recommended_action"],
            title="Deployment Readiness vs Governance Risk",
            labels={"deployment_readiness_score": "Deployment Readiness", "risk_score": "Risk Score"},
            template="plotly_dark",
        )
        fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Decision Signals")
        signal_data = pd.DataFrame(
            [
                ["Best Overall", f"{winners['best_overall']['model_name']} {winners['best_overall']['version']}", winners["best_overall"]["recommended_action"]],
                ["Safest Model", f"{winners['safest_model']['model_name']} {winners['safest_model']['version']}", f"Risk {winners['safest_model']['risk_score']}"],
                ["Fastest Model", f"{winners['fastest_model']['model_name']} {winners['fastest_model']['version']}", f"{winners['fastest_model']['latency_ms']} ms"],
                ["Highest Risk", f"{winners['highest_risk']['model_name']} {winners['highest_risk']['version']}", winners["highest_risk"]["recommended_action"]],
            ],
            columns=["Signal", "Model", "Reason"],
        )
        st.dataframe(signal_data, hide_index=True, use_container_width=True)

        incident_counts = incidents.groupby("severity").size().reset_index(name="count")
        fig = px.bar(incident_counts, x="severity", y="count", title="Open Incident Pressure", template="plotly_dark")
        fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def render_registry(scores: pd.DataFrame) -> None:
    st.subheader("Model Registry And Deployment Gates")
    table = scores[
        [
            "model_name",
            "version",
            "model_type",
            "owner",
            "use_case",
            "deployment_stage",
            "production_status",
            "health_score",
            "risk_status",
            "deployment_readiness_score",
            "recommended_action",
        ]
    ].sort_values("deployment_readiness_score", ascending=False)
    st.dataframe(table, hide_index=True, use_container_width=True)

    fig = px.bar(
        scores.sort_values("deployment_readiness_score", ascending=True),
        x="deployment_readiness_score",
        y="model_name",
        color="recommended_action",
        orientation="h",
        title="Deployment Gate Outcome By Model",
        template="plotly_dark",
    )
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_performance(scores: pd.DataFrame) -> None:
    st.subheader("Performance Comparison")
    selected_metrics = st.multiselect(
        "Choose Metrics To Compare",
        ["accuracy", "precision", "recall", "f1", "confidence_score"],
        default=["accuracy", "precision", "recall", "f1"],
    )
    long_df = scores.melt(id_vars=["model_name", "version"], value_vars=selected_metrics, var_name="metric", value_name="score")
    long_df["model"] = long_df["model_name"] + " " + long_df["version"]

    fig = px.bar(
        long_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Model Quality Metrics",
        template="plotly_dark",
    )
    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latency And Error Exposure")
    col1, col2 = st.columns(2)
    with col1:
        fig_latency = px.bar(scores, x="model_name", y="latency_ms", color="risk_status", title="Latency By Model", template="plotly_dark")
        fig_latency.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
        st.plotly_chart(fig_latency, use_container_width=True)
    with col2:
        fig_error = px.bar(scores, x="model_name", y="error_rate_pct", color="risk_status", title="Estimated Error Rate", template="plotly_dark")
        fig_error.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
        st.plotly_chart(fig_error, use_container_width=True)


def render_drift(scores: pd.DataFrame) -> None:
    st.subheader("Drift And Reliability Monitor")
    drift_df = scores[["model_name", "version", "training_f1", "current_f1", "data_drift_pct", "prediction_drift_pct", "drift_severity"]].copy()
    drift_df["model"] = drift_df["model_name"] + " " + drift_df["version"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Training F1", x=drift_df["model"], y=drift_df["training_f1"] * 100))
    fig.add_trace(go.Bar(name="Current F1", x=drift_df["model"], y=drift_df["current_f1"] * 100))
    fig.update_layout(
        title="Training F1 vs Current Production F1",
        barmode="group",
        template="plotly_dark",
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-25,
        yaxis_title="F1 Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_drift = px.bar(
            drift_df.sort_values("drift_severity", ascending=False),
            x="model",
            y="drift_severity",
            title="Calculated Drift Severity",
            template="plotly_dark",
        )
        fig_drift.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
        st.plotly_chart(fig_drift, use_container_width=True)
    with col2:
        st.dataframe(
            drift_df[["model", "data_drift_pct", "prediction_drift_pct", "drift_severity"]].sort_values("drift_severity", ascending=False),
            hide_index=True,
            use_container_width=True,
        )


def render_risk(scores: pd.DataFrame) -> None:
    st.subheader("AI Risk Scoring Heatmap")
    matrix = build_risk_matrix(scores)
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Risk Matrix Across Drift, Security, Compliance, LLM Safety, And Latency",
        template="plotly_dark",
    )
    fig.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        scores[["model_name", "version", "risk_score", "risk_status", "recommended_action"]].sort_values("risk_score", ascending=False),
        hide_index=True,
        use_container_width=True,
    )


def render_shadow(scores: pd.DataFrame, shadow: pd.DataFrame) -> None:
    st.subheader("Shadow Deployment Review")
    st.caption("Compare current production models against candidate versions before a wider rollout.")
    shadow_summary = build_shadow_summary(shadow, scores)
    st.dataframe(shadow_summary, hide_index=True, use_container_width=True)

    fig = px.bar(
        shadow_summary,
        x="metric",
        y=["current_value", "candidate_value"],
        facet_col="candidate_model",
        barmode="group",
        title="Current vs Candidate Shadow Deployment Results",
        template="plotly_dark",
    )
    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_llm_safety(scores: pd.DataFrame) -> None:
    st.subheader("LLM Safety Evaluation")
    llm_scores = scores[scores["model_type"] == "LLM"].copy()
    if llm_scores.empty:
        st.info("No LLM models are currently in the filtered view.")
        return

    llm_table = llm_scores[
        [
            "model_name",
            "version",
            "hallucination_risk_pct",
            "prompt_injection_risk_pct",
            "pii_leakage_risk_pct",
            "security_exposure_pct",
            "compliance_risk_pct",
            "recommended_action",
        ]
    ].sort_values("security_exposure_pct", ascending=False)
    st.dataframe(llm_table, hide_index=True, use_container_width=True)

    safety_long = llm_scores.melt(
        id_vars=["model_name", "version"],
        value_vars=["hallucination_risk_pct", "prompt_injection_risk_pct", "pii_leakage_risk_pct", "security_exposure_pct"],
        var_name="safety_dimension",
        value_name="risk_value",
    )
    safety_long["model"] = safety_long["model_name"] + " " + safety_long["version"]
    fig = px.bar(
        safety_long,
        x="model",
        y="risk_value",
        color="safety_dimension",
        barmode="group",
        title="LLM Safety Risk Breakdown",
        template="plotly_dark",
    )
    fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)


def render_business_impact(scores: pd.DataFrame) -> None:
    st.subheader("Business Impact Simulator")
    model_names = scores["model_name"].tolist()
    selected = st.selectbox("Select Model", model_names)
    row = scores[scores["model_name"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Monthly Predictions", f"{int(row['monthly_predictions']):,}", "Estimated production volume")
    with c2:
        metric_card("Bad Decisions", f"{int(row['estimated_bad_decisions']):,}", "Based on current error rate")
    with c3:
        metric_card("Monthly Risk", format_currency(row["estimated_monthly_risk_usd"]), "Estimated business exposure")
    with c4:
        metric_card("Monthly AI Cost", format_currency(row["monthly_cost_usd"]), "Operating cost estimate")

    st.markdown("### Operational Recommendation")
    st.markdown(
        f"<div class='panel'><b>{row['model_name']} {row['version']}</b> is currently marked as <b>{row['recommended_action']}</b>. "
        f"The model has a health score of <b>{row['health_score']}</b>, risk score of <b>{row['risk_score']}</b>, "
        f"and an estimated monthly risk exposure of <b>{format_currency(row['estimated_monthly_risk_usd'])}</b>.</div>",
        unsafe_allow_html=True,
    )

    impact_df = scores[["model_name", "estimated_bad_decisions", "estimated_monthly_risk_usd", "monthly_cost_usd", "cost_efficiency_score"]].sort_values("estimated_monthly_risk_usd", ascending=False)
    st.dataframe(impact_df, hide_index=True, use_container_width=True)


def render_incidents(scores: pd.DataFrame, incidents: pd.DataFrame) -> None:
    st.subheader("AI Incident Response Feed")
    merged = incidents.merge(scores[["model_id", "model_name", "version", "owner"]], on="model_id", how="left")
    priority = incident_priority_summary(merged)
    st.dataframe(
        priority[["incident_id", "severity", "model_name", "version", "owner", "issue", "business_impact", "recommended_action", "status"]],
        hide_index=True,
        use_container_width=True,
    )

    fig = px.timeline(
        priority.assign(start="2026-05-01", finish="2026-05-08"),
        x_start="start",
        x_end="finish",
        y="incident_id",
        color="severity",
        hover_name="issue",
        title="Active AI Governance Incident Window",
        template="plotly_dark",
    )
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_governance(scores: pd.DataFrame, audit: pd.DataFrame) -> None:
    st.subheader("Governance Audit Summary")
    merged = audit.merge(scores[["model_id", "model_name", "version", "risk_score", "deployment_readiness_score"]], on="model_id", how="left")
    st.dataframe(
        merged[["audit_id", "model_name", "version", "reviewer", "decision", "review_date", "risk_score", "deployment_readiness_score", "notes"]],
        hide_index=True,
        use_container_width=True,
    )

    fig = px.bar(
        merged,
        x="decision",
        y="risk_score",
        color="reviewer",
        hover_name="model_name",
        title="Governance Decision vs Model Risk",
        template="plotly_dark",
    )
    fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    apply_styles()
    registry, metrics, incidents, audit, shadow = load_data()
    scores = calculate_model_scores(registry, metrics)
    filtered_scores = filter_scores(scores)

    render_header()
    st.divider()

    if filtered_scores.empty:
        st.warning("No models match the current filters. Adjust the sidebar filters to continue.")
        return

    tabs = st.tabs(
        [
            "Executive",
            "Registry",
            "Performance",
            "Drift",
            "Risk Heatmap",
            "Shadow Deployments",
            "LLM Safety",
            "Business Impact",
            "Incidents",
            "Audit Trail",
        ]
    )

    with tabs[0]:
        render_executive_view(filtered_scores, incidents)
    with tabs[1]:
        render_registry(filtered_scores)
    with tabs[2]:
        render_performance(filtered_scores)
    with tabs[3]:
        render_drift(filtered_scores)
    with tabs[4]:
        render_risk(filtered_scores)
    with tabs[5]:
        render_shadow(filtered_scores, shadow)
    with tabs[6]:
        render_llm_safety(filtered_scores)
    with tabs[7]:
        render_business_impact(filtered_scores)
    with tabs[8]:
        render_incidents(filtered_scores, incidents)
    with tabs[9]:
        render_governance(filtered_scores, audit)


if __name__ == "__main__":
    main()

