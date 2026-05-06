from __future__ import annotations

import pandas as pd


def _clip(series: pd.Series, lower: float = 0, upper: float = 100) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def _risk_label(value: float) -> str:
    if value >= 80:
        return "Critical"
    if value >= 60:
        return "High"
    if value >= 35:
        return "Medium"
    return "Low"


def _deployment_action(row: pd.Series) -> str:
    if row["risk_score"] >= 78 or row["security_exposure_pct"] >= 82 or row["compliance_risk_pct"] >= 78:
        return "Blocked By Governance"
    if row["data_drift_pct"] >= 25 or row["prediction_drift_pct"] >= 25 or row["f1_drop_pct"] >= 12:
        return "Retrain Before Release"
    if row["deployment_readiness_score"] >= 85 and row["risk_score"] < 35:
        return "Approved For Production"
    if row["deployment_readiness_score"] >= 70 and row["risk_score"] < 55:
        return "Approved For Shadow Testing"
    if row["deployment_readiness_score"] < 55:
        return "Rollback Recommended"
    return "Risk Team Review"


def calculate_model_scores(registry: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Merge model registry and evaluation telemetry, then calculate enterprise-style scores."""
    df = registry.merge(metrics, on="model_id", how="inner")

    for col in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "training_f1",
        "current_f1",
        "confidence_score",
    ]:
        df[col] = df[col].astype(float)

    df["f1_drop_pct"] = _clip((df["training_f1"] - df["current_f1"]) * 100)
    df["latency_risk"] = _clip(((df["latency_ms"].astype(float) - 250) / 950) * 100)
    df["error_risk"] = _clip(df["error_rate_pct"].astype(float) * 8)

    df["quality_score"] = _clip(
        (
            df["accuracy"] * 0.25
            + df["precision"] * 0.15
            + df["recall"] * 0.15
            + df["f1"] * 0.25
            + df["confidence_score"] * 0.20
        )
        * 100
    )

    df["drift_severity"] = _clip(
        df["data_drift_pct"].astype(float) * 1.4
        + df["prediction_drift_pct"].astype(float) * 1.6
        + df["f1_drop_pct"] * 2.0
    )
    df["drift_health_score"] = _clip(100 - df["drift_severity"])

    df["risk_score"] = _clip(
        df["drift_severity"] * 0.22
        + df["bias_risk_pct"].astype(float) * 0.14
        + df["compliance_risk_pct"].astype(float) * 0.15
        + df["security_exposure_pct"].astype(float) * 0.15
        + df["hallucination_risk_pct"].astype(float) * 0.10
        + df["prompt_injection_risk_pct"].astype(float) * 0.09
        + df["pii_leakage_risk_pct"].astype(float) * 0.08
        + df["latency_risk"] * 0.04
        + df["error_risk"] * 0.03
    )

    df["health_score"] = _clip(
        df["quality_score"] * 0.48 + df["drift_health_score"] * 0.22 + (100 - df["risk_score"]) * 0.30
    )

    df["cost_efficiency_score"] = _clip(
        df["health_score"] / (df["monthly_cost_usd"].astype(float) / 900)
    )

    df["deployment_readiness_score"] = _clip(
        df["quality_score"] * 0.40
        + df["drift_health_score"] * 0.20
        + (100 - df["risk_score"]) * 0.25
        + df["cost_efficiency_score"] * 0.15
    )

    df["estimated_bad_decisions"] = (df["monthly_predictions"].astype(int) * (df["error_rate_pct"].astype(float) / 100)).round(0).astype(int)
    df["estimated_monthly_risk_usd"] = (
        df["estimated_bad_decisions"] * 7.5 + df["risk_score"] * 120
    ).round(0).astype(int)

    df["risk_status"] = df["risk_score"].apply(_risk_label)
    df["recommended_action"] = df.apply(_deployment_action, axis=1)

    score_columns = [
        "quality_score",
        "drift_health_score",
        "risk_score",
        "health_score",
        "cost_efficiency_score",
        "deployment_readiness_score",
        "drift_severity",
        "f1_drop_pct",
        "latency_risk",
        "error_risk",
    ]
    df[score_columns] = df[score_columns].round(1)

    return df

