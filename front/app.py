import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from front.utils import get_predictions, plot_predictions, error_statistics

# Set page config
st.set_page_config(
    page_title="Power Price Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Power Price Forecasting")
st.markdown("""
This application predicts day-ahead electricity prices. Select a target date to see the actual and forecasted prices.
""")

# Date input
target_date = st.date_input(
    "Select Target Date",
    min_value=datetime(2015, 1, 1).date(),
    value=(datetime.today() - timedelta(days=1)).date(),
    help="Select a date to get price predictions"
)

# Add a button to trigger the prediction
if st.button("Get Predictions"):
    with st.spinner("Fetching predictions..."):
        predictions = get_predictions(target_date)
        
        if predictions:
            # Get the plot
            fig = plot_predictions(predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate statistics
                stats = error_statistics(predictions)
                if stats:
                    # Display prediction details
                    st.markdown("### Prediction Details")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Average prices
                    with col1:
                        st.metric("Average Predicted Price", 
                                f"€{stats['avg_predicted']:.2f}/MWh")
                    with col2:
                        st.metric("Average Actual Price", 
                                f"€{stats['avg_actual']:.2f}/MWh")
                    
                    # Error metrics
                    with col3:
                        st.metric("RMSE", f"€{stats['rmse']:.2f}/MWh",
                                help="Root Mean Square Error - Lower values indicate better predictions")
                    with col4:
                        st.metric("MAE", f"€{stats['mae']:.2f}/MWh",
                                help="Mean Absolute Error - Lower values indicate better predictions")
