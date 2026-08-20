import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

def first_digit(num):
    """Extract the first digit from a positive number."""
    num = abs(float(num))
    if num == 0:
        return np.nan

    while num >= 10:
        num /= 10
    while num < 1:
        num *= 10

    return int(num)

def benford_expected():
    """Expected Benford first-digit probabilities for digits 1-9."""
    return {d: np.log10(1 + 1 / d) for d in range(1, 10)}

def observed_first_digit_counts(df, col="amount"):
    """Count observed first digits in a DataFrame column."""
    digits = df[col].apply(first_digit)
    return digits.value_counts().reindex(range(1, 10), fill_value=0).to_dict()

def chi_square_test(observed, expected, n):
    """Run chi-square goodness-of-fit test."""
    obs = np.array([observed.get(d, 0) for d in range(1, 10)], dtype=float)
    exp = np.array([expected[d] * n for d in range(1, 10)], dtype=float)
    res = scipy_stats.chisquare(obs, exp)
    return res.statistic, res.pvalue

def mean_absolute_deviation(observed, expected, n):
    """Mean absolute deviation between observed and expected proportions."""
    obs = np.array([observed.get(d, 0) / n for d in range(1, 10)])
    exp = np.array([expected[d] for d in range(1, 10)])
    return np.mean(np.abs(obs - exp))

def flag_digits(observed, expected, n, z_threshold=2.0):
    """Return digits with positive z-scores above threshold and all z-scores."""
    z_scores = {}
    flagged = []

    for d in range(1, 10):
        exp_count = expected[d] * n
        std = np.sqrt(n * expected[d] * (1 - expected[d]))
        z = (observed.get(d, 0) - exp_count) / std if std > 0 else 0.0
        z_scores[d] = z

        if z > z_threshold:
            flagged.append(d)

    return flagged, z_scores

def flag_transactions(df, col="amount", z_threshold=2.0):
    """Flag transactions whose first digit is significantly overrepresented."""
    n = len(df)
    expected = benford_expected()
    observed = observed_first_digit_counts(df, col)

    flagged_digits, z_scores = flag_digits(observed, expected, n, z_threshold)
    chi2, p_value = chi_square_test(observed, expected, n)
    mad = mean_absolute_deviation(observed, expected, n)

    df = df.copy()
    df["first_digit"] = df[col].apply(first_digit)
    df["is_high_risk"] = df["first_digit"].isin(flagged_digits)

    results = {
        "observed": observed,
        "expected": expected,
        "flagged_digits": flagged_digits,
        "z_scores": z_scores,
        "chi2": chi2,
        "p_value": p_value,
        "mad": mad,
    }

    return df, results

if __name__ == "__main__":
    from data import generate_data

    df = generate_data()
    flagged_df, results = flag_transactions(df)

    total = len(df)
    flagged = flagged_df["is_high_risk"].sum()

    print(f"Total transactions: {total}")
    print(f"Flagged high-risk: {flagged} ({flagged / total:.2%})")
    print(f"Manual review reduction: {1 - flagged / total:.2%}")
    print(f"Chi-square: {results['chi2']:.2f}")
    print(f"p-value: {results['p_value']:.2e}")
    print(f"MAD: {results['mad']:.4f}")
    print(f"Flagged digits: {results['flagged_digits']}")

    if "is_anomaly" in df:
        tp = ((flagged_df["is_high_risk"]) & (flagged_df["is_anomaly"])).sum()
        precision = tp / flagged if flagged else 0
        recall = tp / df["is_anomaly"].sum()
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")