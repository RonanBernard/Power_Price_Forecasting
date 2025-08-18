"""Service for handling ENTSOE data downloads and processing for ATT model."""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
import numpy as np

load_dotenv()

class EntsoeService:
    def __init__(self):
        self.api_key = os.getenv("ENTSOE_API_KEY")
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY not found in environment variables")
        self.client = EntsoePandasClient(api_key=self.api_key)
        
        # For France only as per ATT model
        self.country_code = 'FR'
        
    def get_data_for_date(self, target_date: datetime) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Get required data from ENTSOE for the ATT model.
        
        Args:
            target_date: The target date for prediction
            
        Returns:
            Tuple of (past_data, future_data) dictionaries containing DataFrames
        """
        # Calculate time ranges
        past_start = target_date - timedelta(days=7)  # 7 days of history
        past_end = target_date
        future_start = target_date
        future_end = target_date + timedelta(days=1)  # 24 hours ahead
        
        past_data = {}
        future_data = {}
        
        try:
            # Get past data (7 days)
            past_data['prices'] = self.client.query_day_ahead_prices(
                self.country_code, start=past_start, end=past_end)
            
            past_data['load_forecast'] = self.client.query_load_forecast(
                self.country_code, start=past_start, end=past_end)
            
            past_data['wind_forecast'] = self.client.query_wind_and_solar_forecast(
                self.country_code, start=past_start, end=past_end, psr_type="B19")
            
            past_data['solar_forecast'] = self.client.query_wind_and_solar_forecast(
                self.country_code, start=past_start, end=past_end, psr_type="B16")
            
            # Get future data (next 24 hours)
            future_data['load_forecast'] = self.client.query_load_forecast(
                self.country_code, start=future_start, end=future_end)
            
            future_data['wind_forecast'] = self.client.query_wind_and_solar_forecast(
                self.country_code, start=future_start, end=future_end, psr_type="B19")
            
            future_data['solar_forecast'] = self.client.query_wind_and_solar_forecast(
                self.country_code, start=future_start, end=future_end, psr_type="B16")
            
        except Exception as e:
            raise Exception(f"Error fetching ENTSOE data: {e}")
        
        return past_data, future_data

    def process_raw_data(self, past_data: Dict[str, pd.DataFrame], 
                        future_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process raw ENTSOE data into the format required by the ATT model.
        
        Args:
            past_data: Dictionary of past raw data from ENTSOE
            future_data: Dictionary of future raw data from ENTSOE
            
        Returns:
            Tuple of (past_df, future_df) processed and aligned DataFrames
        """
        # Process past data
        past_df = pd.DataFrame()
        past_df['price'] = past_data['prices']
        past_df['load_forecast'] = past_data['load_forecast']
        past_df['wind_forecast'] = past_data['wind_forecast']
        past_df['solar_forecast'] = past_data['solar_forecast']
        
        # Process future data
        future_df = pd.DataFrame()
        future_df['load_forecast'] = future_data['load_forecast']
        future_df['wind_forecast'] = future_data['wind_forecast']
        future_df['solar_forecast'] = future_data['solar_forecast']
        
        # Handle missing values with forward fill then backward fill
        past_df = past_df.fillna(method='ffill').fillna(method='bfill')
        future_df = future_df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all required columns are present
        required_past_cols = ['price', 'load_forecast', 'wind_forecast', 'solar_forecast']
        required_future_cols = ['load_forecast', 'wind_forecast', 'solar_forecast']
        
        missing_past = set(required_past_cols) - set(past_df.columns)
        missing_future = set(required_future_cols) - set(future_df.columns)
        
        if missing_past:
            raise ValueError(f"Missing required past features: {missing_past}")
        if missing_future:
            raise ValueError(f"Missing required future features: {missing_future}")
        
        return past_df, future_df
