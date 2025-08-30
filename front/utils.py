import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests

API_URL = "https://power-da-price-1040927723543.europe-west1.run.app/api/v1/predictions/predict"  # API endpoint with correct path

def format_date_for_api(date):
    """Convert datetime to DD/MM/YYYY format for API"""
    return date.strftime("%d/%m/%Y")

def get_predictions(target_date):
    """Fetch predictions from the API"""
    try:
        formatted_date = format_date_for_api(target_date)
        st.write(f"Making request to {API_URL} with date: {formatted_date}")
        
        response = requests.post(
            API_URL,
            json={"date": formatted_date}
        )
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def error_statistics(predictions):
    """Calculate error statistics between predicted and actual values"""
    if not predictions or not any(price is not None for price in predictions["actual_prices"]):
        return None
    
    predicted = predictions["predicted_prices"]
    actual = predictions["actual_prices"]
    
    # Filter out None values from actual prices
    valid_pairs = [(p, a) for p, a in zip(predicted, actual) if a is not None]
    if not valid_pairs:
        return None
        
    predicted_valid, actual_valid = zip(*valid_pairs)
    
    # Calculate statistics
    stats = {}
    
    # Basic statistics
    stats["avg_predicted"] = sum(predicted) / len(predicted)
    stats["avg_actual"] = sum(filter(None, actual)) / len(list(filter(None, actual)))
    
    # Error metrics
    # RMSE
    mse = sum((p - a) ** 2 for p, a in zip(predicted_valid, actual_valid)) / len(predicted_valid)
    stats["rmse"] = (mse) ** 0.5
    
    # MAE
    stats["mae"] = sum(abs(p - a) for p, a in zip(predicted_valid, actual_valid)) / len(predicted_valid)
    
    return stats

def plot_predictions(predictions):
    """Create a plot comparing actual and predicted prices"""
    if not predictions:
        return None
    
    # Create time points for x-axis (24 hours)
    hours = list(range(24))
    
    # Create the figure
    fig = go.Figure()
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=hours,
        y=predictions["predicted_prices"],
        name="Predicted Prices",
        line=dict(color="#1f77b4", width=2)
    ))
    
    # Add actual prices if available
    if any(price is not None for price in predictions["actual_prices"]):
        fig.add_trace(go.Scatter(
            x=hours,
            y=predictions["actual_prices"],
            name="Actual Prices",
            line=dict(color="#2ca02c", width=2, dash="dash")
        ))
    
    # Update layout
    fig.update_layout(
        title="Power Price Forecast",
        xaxis_title="Hour of Day",
        yaxis_title="Price (€/MWh)",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig