from __future__ import annotations

import pandas as pd


RISK_COLUMNS = [
    "risk_score",
    "drift_severity",
    "bias_risk_pct",
    "security_exposure_pct",
    "compliance_risk_pct",
    "hallucination_risk_pct",
    "prompt_injection_risk_pct",
    "pii_leakage_risk_pct",
    "latency_risk",
    "error_risk",
]


def build_risk_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    matrix = scores[["model_name", "version", *RISK_COLUMNS]].copy()
    matrix["model"] = matrix["model_name"] + " " + matrix["version"]
    return matrix.set_index("model")[RISK_COLUMNS]


def incident_priority_summary(incidents: pd.DataFrame) -> pd.DataFrame:
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    data = incidents.copy()
    data["severity_rank"] = data["severity"].map(severity_order).fillna(4)
    return data.sort_values(["severity_rank", "status", "incident_id"]).drop(columns=["severity_rank"])

