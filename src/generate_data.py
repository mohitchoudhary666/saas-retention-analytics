from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PLANS = ("Starter", "Growth", "Business", "Enterprise")
REGIONS = ("North America", "Europe", "Asia Pacific", "Other")


def generate_subscriptions(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create fictional subscription records with reproducible churn patterns."""
    if rows < 100:
        raise ValueError("Generate at least 100 rows so the dashboard segments are useful.")
    rng = np.random.default_rng(seed)
    today = pd.Timestamp("2026-09-30")
    signup_days_ago = rng.integers(30, 1095, size=rows)
    signup_date = today - pd.to_timedelta(signup_days_ago, unit="D")
    plan = rng.choice(PLANS, size=rows, p=[0.43, 0.32, 0.18, 0.07])
    plan_price = pd.Series(plan).map({"Starter": 29, "Growth": 89, "Business": 249, "Enterprise": 799}).to_numpy()
    expected_seats = np.select(
        [plan == "Starter", plan == "Growth", plan == "Business", plan == "Enterprise"],
        [2, 6, 18, 60], default=2,
    )
    seats = np.maximum(1, rng.poisson(expected_seats))
    monthly_revenue = (plan_price + np.maximum(seats - 1, 0) * 8).astype(float)
    region = rng.choice(REGIONS, size=rows, p=[0.43, 0.26, 0.23, 0.08])
    tenure_months = np.maximum(1, (signup_days_ago / 30.4375).astype(int))
    support_tickets_90d = rng.poisson(0.7 + (plan == "Enterprise") * 0.8, size=rows)
    weekly_active_days = np.clip(rng.normal(3.6, 1.8, size=rows), 0, 7).round(1)
    feature_adoption_pct = np.clip(rng.beta(2.2, 2.1, size=rows) * 100, 2, 100).round(1)
    payment_failures_12m = rng.poisson(0.15, size=rows)

    # Synthetic labels are intentionally generated from transparent risk factors.
    logit = (
        -2.55
        + 0.30 * (weekly_active_days < 1.5)
        + 0.75 * (feature_adoption_pct < 20)
        + 0.37 * (support_tickets_90d >= 3)
        + 0.95 * (payment_failures_12m > 0)
        - 0.035 * np.minimum(tenure_months, 24)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_probability).astype(bool)
    max_cancel_days = np.maximum(1, signup_days_ago - 1)
    cancel_days_ago = np.minimum(rng.integers(1, 180, size=rows), max_cancel_days)
    cancel_date = pd.Series(pd.NaT, index=np.arange(rows), dtype="datetime64[ns]")
    cancel_date.loc[churned] = today - pd.to_timedelta(cancel_days_ago[churned], unit="D")

    frame = pd.DataFrame({
        "customer_id": [f"CUS-{index:05d}" for index in range(1, rows + 1)],
        "signup_date": signup_date,
        "cancel_date": cancel_date,
        "status": np.where(churned, "Churned", "Active"),
        "plan": plan,
        "region": region,
        "seats": seats,
        "monthly_revenue_usd": monthly_revenue.round(2),
        "tenure_months": tenure_months,
        "weekly_active_days": weekly_active_days,
        "feature_adoption_pct": feature_adoption_pct,
        "support_tickets_90d": support_tickets_90d,
        "payment_failures_12m": payment_failures_12m,
    })
    return frame.sort_values("signup_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/generated/subscriptions.csv"))
    args = parser.parse_args()
    data = generate_subscriptions(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(data):,} synthetic subscriptions to {args.output}")


if __name__ == "__main__":
    main()

