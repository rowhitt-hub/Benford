import pandas as pd
import numpy as np

def get_expected_benford_dist():
   
    digits = np.arange(1, 10)
    # Applying the logarithmic formula: P(d) = log10(1 + 1/d)
    probabilities = np.log10(1 + 1 / digits)
    
    expected_df = pd.DataFrame({
        'Digit': digits,
        'Expected_Prob': probabilities
    })
    return expected_df.set_index('Digit')

def extract_first_digit(series):
   
    # Convert to string and use regex to find the first digit between 1 and 9
    first_digits = series.astype(str).str.extract(r'([1-9])')[0]
    
    # Drop NaN and convert to integer
    return first_digits.dropna().astype(int)

def get_actual_dist(series):
  
    first_digits = extract_first_digit(series)
    
    # Count the occurrences of each digit
    counts = first_digits.value_counts().sort_index()
    
    # Calculate the proportion/probability of each digit
    total_count = counts.sum()
    probabilities = counts / total_count
    
    # Ensure all digits 1-9 are represented even if count is 0
    actual_df = pd.DataFrame({'Actual_Prob': probabilities, 'Count': counts})
    actual_df = actual_df.reindex(range(1, 10), fill_value=0)
    actual_df.index.name = 'Digit'
    
    return actual_df, total_count