"""Service for handling ENTSOE data downloads and processing."""

import os
from datetime import datetime, timedelta
from typing import Dict, List
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
        
        # Define country codes
        self.country_codes = {
            'France': 'FR',
            'Germany': 'DE_LU',  # Post Oct 2018
            'Belgium': 'BE',
            'Switzerland': 'CH',
            'Spain': 'ES',
            'Italy': 'IT_NORD'  # North Italy zone
        }

    def get_data_for_date(self, target_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get all required data from ENTSOE for a specific date.
        
        Args:
            target_date: The target date for prediction
            
        Returns:
            Dictionary containing DataFrames for each data type
        """
        # We need data from previous days for feature engineering
        start = target_date - timedelta(days=7)  # Get 7 days of history
        end = target_date + timedelta(days=1)    # Get target day
        
        data = {}
        
        # Get data for each country (focusing on France for now)
        country = 'France'
        country_code = self.country_codes[country]
        
        try:
            # Get day-ahead prices
            data['prices'] = self.client.query_day_ahead_prices(
                country_code, start=start, end=end)
            
            # Get load forecast
            data['load_forecast'] = self.client.query_load_forecast(
                country_code, start=start, end=end)
            
            # Get actual load
            data['load'] = self.client.query_load(
                country_code, start=start, end=end)
            
            # Get wind and solar forecasts
            data['wind_forecast'] = self.client.query_wind_and_solar_forecast(
                country_code, start=start, end=end, psr_type="B19")  # Wind
            data['solar_forecast'] = self.client.query_wind_and_solar_forecast(
                country_code, start=start, end=end, psr_type="B16")  # Solar
            
            # Get cross-border flows
            neighboring_countries = ['BE', 'CH', 'DE_LU', 'ES', 'IT_NORD']
            flows = []
            for neighbor in neighboring_countries:
                try:
                    flow = self.client.query_crossborder_flows(
                        country_code, neighbor, start=start, end=end)
                    flow.name = f'flow_{country_code}_{neighbor}'
                    flows.append(flow)
                except Exception as e:
                    print(f"Error getting flow data for {neighbor}: {e}")
            
            if flows:
                data['flows'] = pd.concat(flows, axis=1)
            
        except Exception as e:
            raise Exception(f"Error fetching ENTSOE data: {e}")
        
        return data

    def process_raw_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw ENTSOE data into a format suitable for the model.
        
        Args:
            data: Dictionary of raw data from ENTSOE
            
        Returns:
            Processed DataFrame ready for feature engineering
        """
        df = pd.DataFrame()
        
        # Process prices
        if 'prices' in data:
            df['price'] = data['prices']
        
        # Process load
        if 'load_forecast' in data:
            df['load_forecast'] = data['load_forecast']
        if 'load' in data:
            df['load_actual'] = data['load']
        
        # Process renewables forecasts
        if 'wind_forecast' in data:
            df['wind_forecast'] = data['wind_forecast']
        if 'solar_forecast' in data:
            df['solar_forecast'] = data['solar_forecast']
        
        # Process flows
        if 'flows' in data:
            for col in data['flows'].columns:
                df[col] = data['flows'][col]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
