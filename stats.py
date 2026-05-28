import pandas as pd
from scipy.stats import chisquare

def perform_chi_square_test(actual_df, expected_df, total_count):
     
    # 1. Align the data 
    actual_counts = actual_df['Count'].reindex(range(1, 10), fill_value=0).values
    expected_probs = expected_df['Expected_Prob'].reindex(range(1, 10)).values
    
    # 2. Convert expected probabilities to expected counts
    # Chi-Square requires absolute frequencies, not percentages.
    expected_counts = expected_probs * total_count
    
    # 3. Run the statistical test
    chi2_stat, p_value = chisquare(f_obs=actual_counts, f_exp=expected_counts)
    
    return chi2_stat, p_value

def interpret_p_value(p_value, alpha=0.05):

    is_anomalous = p_value < alpha
    
    if is_anomalous:
        summary = (
            f"🚨 **Anomaly Detected:** The dataset significantly deviates from Benford's Law "
            f"(p-value: {p_value:.4e}). This warrants further investigation."
        )
    else:
        summary = (
            f"✅ **Normal:** The dataset conforms to naturally occurring numerical distributions "
            f"(p-value: {p_value:.4f}). No immediate red flags."
        )
        
    return is_anomalous, summary