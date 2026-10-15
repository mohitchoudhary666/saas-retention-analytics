from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {
    "customer_id", "signup_date", "cancel_date", "status", "plan", "region", "seats",
    "monthly_revenue_usd", "tenure_months", "weekly_active_days", "feature_adoption_pct",
    "support_tickets_90d", "payment_failures_12m",
}
NUMERIC_FEATURES = [
    "seats", "monthly_revenue_usd", "tenure_months", "weekly_active_days",
    "feature_adoption_pct", "support_tickets_90d", "payment_failures_12m",
]
CATEGORICAL_FEATURES = ["plan", "region"]


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["signup_date", "cancel_date"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The dataset has no customer rows.")
    if frame["customer_id"].duplicated().any():
        raise ValueError("customer_id values must be unique.")
    if not frame["status"].isin(["Active", "Churned"]).all():
        raise ValueError("status must contain only Active or Churned values.")
    return frame


def cohort_retention(frame: pd.DataFrame, observation_date: pd.Timestamp) -> pd.DataFrame:
    data = frame[["customer_id", "signup_date", "cancel_date"]].copy()
    data["cohort"] = data["signup_date"].dt.to_period("M").dt.to_timestamp()
    observed_month = observation_date.to_period("M")
    data["age_month"] = data["cohort"].map(
        lambda cohort: max(0, (observed_month - cohort.to_period("M")).n)
    )
    data["cancel_month"] = data["cancel_date"].dt.to_period("M")
    rows: list[dict] = []
    for cohort, group in data.groupby("cohort", observed=True):
        size = len(group)
        cancel_age = group["cancel_month"].map(
            lambda month: (month - cohort.to_period("M")).n if pd.notna(month) else np.nan
        )
        for age in range(int(group["age_month"].max()) + 1):
            retained = int((cancel_age.isna() | (cancel_age > age)).sum())
            rows.append({"cohort": cohort, "month": age, "customers": size,
                         "retention_pct": 100 * retained / size})
    return pd.DataFrame(rows)


def cancellation_counts(frame: pd.DataFrame) -> pd.DataFrame:
    canceled = frame[frame["cancel_date"].notna()].copy()
    canceled["month"] = canceled["cancel_date"].dt.to_period("M").dt.to_timestamp()
    counts = canceled.groupby("month").size().rename("cancellations").reset_index()
    return counts.sort_values("month")


def train_churn_model(frame: pd.DataFrame) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    target = (frame["status"] == "Churned").astype(int)
    if target.nunique() < 2 or target.value_counts().min() < 5:
        raise ValueError("Need at least five active and five churned customers to train the demo model.")
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(test_x)[:, 1]
    predicted = probabilities >= 0.5
    metrics = {
        "ROC-AUC": float(roc_auc_score(test_y, probabilities)),
        "Accuracy": float(accuracy_score(test_y, predicted)),
        "Precision": float(precision_score(test_y, predicted, zero_division=0)),
        "Recall": float(recall_score(test_y, predicted, zero_division=0)),
    }
    active = frame[frame["status"] == "Active"].copy()
    active["churn_risk_pct"] = 100 * pipeline.predict_proba(
        active[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    active["risk_band"] = pd.cut(active["churn_risk_pct"],
                                   bins=[-0.1, 34, 67, 100],
                                   labels=["Low", "Medium", "High"])
    return pipeline, metrics, active.sort_values("churn_risk_pct", ascending=False)


def coefficient_importance(pipeline: Pipeline) -> pd.DataFrame:
    encoder = pipeline.named_steps["features"].named_transformers_["categorical"]
    names = NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    result = pd.DataFrame({"feature": names, "coefficient": pipeline.named_steps["model"].coef_[0]})
    result["direction"] = np.where(result["coefficient"] > 0,
                                    "Higher modeled risk", "Lower modeled risk")
    return result.reindex(result["coefficient"].abs().sort_values(ascending=False).index)

