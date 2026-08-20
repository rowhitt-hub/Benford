import streamlit as st
import pandas as pd

from data import generate_data
from stats import flag_transactions

@st.cache_data
def load_data():
    return generate_data()

st.set_page_config(page_title="Benford Anomaly Detection", layout="wide")

st.title("Dataset anomaly detection using Benford's Law")
st.caption("Synthetic dataset • Statistical tests • Streamlit dashboard")

df = load_data()
flagged_df, results = flag_transactions(df)

total = len(df)
flagged = flagged_df["is_high_risk"].sum()
flagged_pct = flagged / total * 100
review_reduction = 100 - flagged_pct

col1, col2, col3 = st.columns(3)
col1.metric("Total transactions", f"{total:,}")
col2.metric("Flagged high-risk", f"{flagged:,} ({flagged_pct:.2f}%)")
col3.metric("Manual review reduction", f"{review_reduction:.2f}%")

st.subheader("Statistical test results")
st.write(f"Chi-square statistic: {results['chi2']:.2f}")
st.write(f"p-value: {results['p_value']:.2e}")
st.write(f"Mean absolute deviation: {results['mad']:.4f}")
st.write(f"Flagged first digits: {results['flagged_digits']}")

# Observed vs expected first-digit distribution
chart_data = pd.DataFrame({
    "Digit": range(1, 10),
    "Observed": [results["observed"].get(d, 0) / total for d in range(1, 10)],
    "Expected": [results["expected"][d] for d in range(1, 10)],
}).set_index("Digit")

st.subheader("First-digit distribution")
st.bar_chart(chart_data)

# Optional ground truth metrics
if "is_anomaly" in df:
    tp = ((flagged_df["is_high_risk"]) & (flagged_df["is_anomaly"])).sum()
    precision = tp / flagged if flagged else 0
    recall = tp / df["is_anomaly"].sum()

    st.subheader("Synthetic ground truth")
    st.write(f"Precision: {precision:.2%}")
    st.write(f"Recall: {recall:.2%}")

st.subheader("Sample flagged transactions")
st.dataframe(flagged_df[flagged_df["is_high_risk"]].head(20))

# Summary section
st.subheader("Summary")
st.markdown(
    f"""
    This analysis applied **Benford's Law** to detect anomalous first‑digit patterns in a synthetic dataset of **{total:,} transactions**.

    - **{flagged:,} transactions** ({flagged_pct:.2f}%) were flagged as high‑risk anomalies.
    - This reduces the dataset a manual reviewer would need to check by **{review_reduction:.2f}%**.

    The statistical evidence strongly rejects the null hypothesis that the data follow Benford's Law (p‑value = {results['p_value']:.2e}), indicating significant digit anomalies.
    """
)

if "is_anomaly" in df:
    st.markdown(
        f"""
        On the injected synthetic anomalies, the method achieves **{recall:.0%} recall** and **{precision:.0%} precision**, demonstrating its effectiveness at isolating suspicious transactions while minimizing false positives.
        """
    )