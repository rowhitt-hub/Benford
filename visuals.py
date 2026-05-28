"""
    Creates a bar chart of the actual distribution overlaid with a 
    line graph of the expected Benford's Law distribution.
    Returns the matplotlib figure for Streamlit to render.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_benford_distribution(actual_df, expected_df):
  
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    digits = actual_df.index
    actual_probs = actual_df['Actual_Prob']
    expected_probs = expected_df['Expected_Prob']
    
    # 1. Bar chart for the actual data
    ax.bar(digits, actual_probs, color='#4C72B0', edgecolor='black', 
           alpha=0.7, label='Actual PaySim Data')
    
    # 2. Line chart for the theoretical Benford's Law
    ax.plot(digits, expected_probs, color='#C44E52', marker='o', 
            linestyle='-', linewidth=2.5, markersize=8, 
            label="Benford's Law (Expected)")
    
    # Formatting the aesthetics
    ax.set_title("Leading Digit Distribution vs. Benford's Law", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Leading Digit", fontsize=12, fontweight='bold')
    ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
    ax.set_xticks(digits)
    
    # Convert y-axis to readable percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Clean up the grid and legend
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Remove top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # tight_layout prevents labels from being cut off in Streamlit
    fig.tight_layout()
    
    return fig

def plot_digit_deviations(actual_df, expected_df):
    """
    Plots the absolute percentage difference between actual and expected 
    distributions to highlight exactly which digits are anomalous.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    digits = actual_df.index
    # Calculate difference in percentage points
    deviations = (actual_df['Actual_Prob'] - expected_df['Expected_Prob']) * 100 
    
    # Color coding: Red if positive (padding numbers), Blue if negative
    colors = ['#C44E52' if val > 0 else '#4C72B0' for val in deviations]
    
    ax.bar(digits, deviations, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1.5) # The zero-error baseline
    
    # Formatting
    ax.set_title("Deviation from Benford's Law by Digit", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Leading Digit", fontsize=12)
    ax.set_ylabel("Deviation (%)", fontsize=12)
    ax.set_xticks(digits)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    return fig