import streamlit as st
import pandas as pd

# Import your custom modules
import benford
import stats
import visuals

# --- Page Configuration ---
st.set_page_config(page_title="Benford's Law Fraud Detection", layout="wide")
st.title("📊 Benford's Law: Anomalous Ledger Detection")
st.markdown("Upload a financial dataset (like PaySim) to analyze leading digit distributions for statistical anomalies.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
    
    # Select the numerical column to analyze
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    target_col = st.selectbox("Select the column to analyze (e.g., transaction amounts):", numeric_cols)
    
    if st.button("Run Analysis"):
        with st.spinner("Crunching numbers..."):
            
            # 2. Core Logic (benford.py)
            expected_df = benford.get_expected_benford_dist()
            actual_df, total_count = benford.get_actual_dist(df[target_col])
            
            # 3. Math & Rigor (stats.py)
            chi2_stat, p_value = stats.perform_chi_square_test(actual_df, expected_df, total_count)
            is_anomalous, summary_text = stats.interpret_p_value(p_value)
            
            # --- DISPLAY RESULTS ---
            st.divider()
            
            # Print the statistical conclusion
            if is_anomalous:
                st.error(summary_text)
            else:
                st.success(summary_text)
                
            st.markdown(f"**Total Records Analyzed:** {total_count:,}")
            
            # 4. Visualizations (visuals.py)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution Overlay")
                fig1 = visuals.plot_benford_distribution(actual_df, expected_df)
                st.pyplot(fig1)
                
            with col2:
                st.subheader("Deviation Analysis")
                fig2 = visuals.plot_digit_deviations(actual_df, expected_df)
                st.pyplot(fig2)
                
            # Optional: Show the raw data table
            with st.expander("View Raw Distribution Data"):
                # Combine actual and expected for easy reading
                summary_df = pd.concat([actual_df['Actual_Prob'], expected_df['Expected_Prob']], axis=1)
                summary_df.columns = ['Actual Probability', 'Expected Probability']
                st.dataframe(summary_df.style.format("{:.4%}") )