import numpy as np
import pandas as pd

def generate_data(n_total=50_000, n_anomalies=2_500, seed=42):
    """Generate synthetic transaction data.

    Most amounts follow a log-uniform distribution, so first digits
    naturally follow Benford's Law. Anomalies are concentrated in
    the 900-999.99 range, creating an overrepresentation of digit 9.
    """
    rng = np.random.default_rng(seed)
    n_normal = n_total - n_anomalies

    # Normal amounts: log-uniform -> Benford-compatible first digits
    normal_amounts = np.round(
        10 ** rng.uniform(np.log10(1), np.log10(100_000), n_normal),
        2,
    )

    # Anomalies: concentrated in first-digit 9
    anomaly_amounts = np.round(rng.uniform(900, 999.99, n_anomalies), 2)

    amounts = np.concatenate([normal_amounts, anomaly_amounts])
    is_anomaly = np.concatenate([
        np.zeros(n_normal, dtype=bool),
        np.ones(n_anomalies, dtype=bool),
    ])

    # Shuffle data
    idx = rng.permutation(n_total)
    amounts = amounts[idx]
    is_anomaly = is_anomaly[idx]

    merchants = rng.choice(
        [
            "Acme Corp",
            "Globex",
            "Initech",
            "Umbrella",
            "Hooli",
            "Stark Industries",
            "Wayne Enterprises",
        ],
        size=n_total,
    )

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_total), unit="D"
    )

    df = pd.DataFrame({
        "transaction_id": [f"TX-{i:05d}" for i in range(n_total)],
        "amount": amounts,
        "merchant": merchants,
        "date": dates,
        "is_anomaly": is_anomaly,
    })

    return df

if __name__ == "__main__":
    df = generate_data()
    print(df.head())
    print(f"\nTotal transactions: {len(df)}")
    print(f"Injected anomalies: {df['is_anomaly'].sum()}")