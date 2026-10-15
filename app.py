from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import cancellation_counts, cohort_retention, coefficient_importance, load_data, train_churn_model
from src.generate_data import generate_subscriptions


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "sample_subscriptions.csv"
OBSERVATION_DATE = pd.Timestamp("2026-09-30")

st.set_page_config(page_title="SaaS Retention Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#f6f8fc 0%,#f0f5fa 45%,#f8f5ff 100%);}
[data-testid="stMetric"] {background:rgba(255,255,255,.82);border:1px solid #e1e8f0;padding:16px;border-radius:14px;box-shadow:0 5px 20px #15304a0b;}
div[data-testid="stSidebar"] {background:#101a2b;}
div[data-testid="stSidebar"] * {color:#e7eef9;}
h1,h2,h3 {letter-spacing:-.025em;}
</style>
""", unsafe_allow_html=True)

st.title("📊 SaaS Retention Lab")
st.caption("Understand subscription retention, compare customer cohorts, and explore transparent churn-risk signals.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a subscriptions CSV", type=["csv"])
    st.caption("Use the documented schema in the README. Your data stays in this local session.")
    st.divider()
    st.markdown("**Dataset**\n\nFictional records only · No AWS account needed")

if uploaded:
    try:
        frame = load_data(uploaded)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not use that CSV: {error}")
        st.stop()
else:
    frame = load_data(DEFAULT_DATA) if DEFAULT_DATA.exists() else generate_subscriptions()

active = frame[frame["status"] == "Active"]
churned = frame[frame["status"] == "Churned"]
total_customers = len(frame)
churn_rate = (len(churned) / total_customers * 100) if total_customers else 0
mrr = active["monthly_revenue_usd"].sum()
median_tenure = frame["tenure_months"].median()

first, second, third, fourth = st.columns(4)
first.metric("Customers", f"{total_customers:,}")
second.metric("Active MRR", f"${mrr:,.0f}", help="Sum of monthly revenue for currently active synthetic subscriptions.")
third.metric("Sample churn rate", f"{churn_rate:.1f}%", help="Churned customer records divided by all records in this sample.")
fourth.metric("Median tenure", f"{median_tenure:.0f} months")

tab_overview, tab_retention, tab_risk, tab_data = st.tabs(["Overview", "Cohort retention", "Churn signals", "Data explorer"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Customers by plan")
        by_plan = frame.groupby(["plan", "status"], observed=True).size().reset_index(name="customers")
        st.plotly_chart(px.bar(by_plan, x="plan", y="customers", color="status", barmode="group",
                               color_discrete_map={"Active": "#22a783", "Churned": "#f07864"}),
                        use_container_width=True)
    with right:
        st.subheader("Customer distribution by region")
        by_region = frame.groupby("region", observed=True).agg(
            customers=("customer_id", "count"),
            churn_rate=("status", lambda values: (values == "Churned").mean() * 100),
        ).reset_index()
        st.plotly_chart(px.bar(by_region, x="region", y="churn_rate", color="customers",
                               labels={"churn_rate": "Churned share (%)"}, color_continuous_scale="Tealgrn"),
                        use_container_width=True)
    st.subheader("Monthly cancellations in the sample")
    counts = cancellation_counts(frame)
    st.plotly_chart(px.bar(counts, x="month", y="cancellations", color="cancellations",
                           labels={"cancellations": "Cancelled subscriptions"}, color_continuous_scale="Oranges"),
                    use_container_width=True)

with tab_retention:
    st.subheader("Signup-cohort retention")
    retention = cohort_retention(frame, OBSERVATION_DATE)
    cohorts = sorted(retention["cohort"].dropna().unique())
    selected_cohorts = st.multiselect("Select signup cohorts", cohorts, default=cohorts[-6:],
                                      format_func=lambda value: pd.Timestamp(value).strftime("%b %Y"))
    view = retention[retention["cohort"].isin(selected_cohorts)]
    if view.empty:
        st.info("Select one or more cohorts to display retention.")
    else:
        st.plotly_chart(px.line(view, x="month", y="retention_pct", color="cohort", markers=True,
                                labels={"month": "Months since signup", "retention_pct": "Retained customers (%)", "cohort": "Signup cohort"},
                                hover_data={"customers": True}), use_container_width=True)
    st.caption("Retention is calculated from each signup month and cancellation date. Recent cohorts have shorter observation windows.")

with tab_risk:
    st.subheader("Explainable churn-risk model")
    st.write("A logistic-regression baseline learns patterns from the synthetic sample. Use it to explore analytics workflows, not make real customer decisions.")
    try:
        model, metrics, scored = train_churn_model(frame)
    except ValueError as error:
        st.warning(str(error))
    else:
        cols = st.columns(4)
        for column, (label, value) in zip(cols, metrics.items(), strict=True):
            column.metric(label, f"{value:.3f}")
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("**Highest-risk active subscriptions**")
            st.dataframe(scored[["customer_id", "plan", "region", "churn_risk_pct", "risk_band",
                                 "weekly_active_days", "feature_adoption_pct", "support_tickets_90d"]].head(20),
                         use_container_width=True, hide_index=True)
        with right:
            importance = coefficient_importance(model).head(10).sort_values("coefficient")
            st.markdown("**Largest model coefficients**")
            st.plotly_chart(px.bar(importance, x="coefficient", y="feature", color="direction", orientation="h",
                                   color_discrete_map={"Higher modeled risk": "#e67c68", "Lower modeled risk": "#22a783"}),
                            use_container_width=True)
        st.caption("Metrics use a stratified 25% holdout split. Synthetic labels are generated from the same demo signals; scores do not establish real-world predictive performance.")

with tab_data:
    st.subheader("Customer records")
    plans = st.multiselect("Filter plans", sorted(frame["plan"].unique()), default=sorted(frame["plan"].unique()))
    statuses = st.multiselect("Filter status", ["Active", "Churned"], default=["Active", "Churned"])
    filtered = frame[frame["plan"].isin(plans) & frame["status"].isin(statuses)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                       "saas-customer-subscriptions.csv", "text/csv")

st.divider()
st.caption("Demo only: data is synthetic; modeled relationships are illustrative and must not be used for real customer decisions.")

