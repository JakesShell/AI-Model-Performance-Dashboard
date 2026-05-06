from __future__ import annotations

import pandas as pd


def get_best_models(scores: pd.DataFrame) -> dict[str, pd.Series]:
    """Return portfolio-level winners and watch-list models."""
    return {
        "best_overall": scores.sort_values(["deployment_readiness_score", "health_score"], ascending=False).iloc[0],
        "safest_model": scores.sort_values(["risk_score", "deployment_readiness_score"], ascending=[True, False]).iloc[0],
        "fastest_model": scores.sort_values(["latency_ms", "health_score"], ascending=[True, False]).iloc[0],
        "highest_risk": scores.sort_values(["risk_score", "estimated_monthly_risk_usd"], ascending=False).iloc[0],
        "most_expensive": scores.sort_values("monthly_cost_usd", ascending=False).iloc[0],
    }


def build_executive_summary(scores: pd.DataFrame) -> str:
    winners = get_best_models(scores)
    approved = scores[scores["recommended_action"].str.contains("Approved", case=False, na=False)]
    blocked = scores[scores["recommended_action"].str.contains("Blocked|Rollback|Retrain", case=False, na=False)]

    best = winners["best_overall"]
    risk = winners["highest_risk"]

    return (
        f"{best['model_name']} {best['version']} is currently the strongest deployment candidate "
        f"with a readiness score of {best['deployment_readiness_score']} and a risk score of {best['risk_score']}. "
        f"{len(approved)} model(s) are cleared for production or shadow testing, while {len(blocked)} model(s) require governance attention. "
        f"The highest-risk model is {risk['model_name']} {risk['version']}, driven by a {risk['risk_status'].lower()} risk rating, "
        f"{risk['data_drift_pct']}% data drift, and an estimated monthly risk exposure of ${risk['estimated_monthly_risk_usd']:,}."
    )


def build_shadow_summary(shadow: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    model_lookup = scores.set_index("model_id")[["model_name", "version", "recommended_action", "risk_score", "deployment_readiness_score"]]
    summary = shadow.copy()
    summary["delta"] = summary["candidate_value"] - summary["current_value"]
    summary["candidate_model"] = summary["candidate_model_id"].map(lambda x: f"{model_lookup.loc[x, 'model_name']} {model_lookup.loc[x, 'version']}" if x in model_lookup.index else x)
    summary["current_model"] = summary["current_model_id"].map(lambda x: f"{model_lookup.loc[x, 'model_name']} {model_lookup.loc[x, 'version']}" if x in model_lookup.index else x)
    return summary

