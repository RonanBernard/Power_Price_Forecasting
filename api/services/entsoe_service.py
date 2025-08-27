"""Service for handling ENTSOE data downloads and processing for ATT model."""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from entsoe import EntsoePandasClient
from scripts.config import (
    ENTSOE_COUNTRY_CODES,
    CROSSBORDER_COUNTRY_CODES,
    GERMANY_HISTORICAL,
    COUNTRY_DICT,
    TIMEZONE
)


def download_entsoe_data(
    api_key: str,
    target_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get required data from ENTSOE for the ATT model.
    
    Args:
        api_key: ENTSOE API key
        target_date: The target date for prediction
        max_workers: Maximum number of parallel workers
        data_types: Optional list of data types to download
        
    Returns:
        Tuple of (past_data, future_data) DataFrames with aligned hourly data
        - past_data contains historical data and fuel prices
        - future_data contains forecasts (wind, solar, load) with datetime index
        - data_target contains actual prices for the target date
    """
    countries = list(ENTSOE_COUNTRY_CODES.keys()) + ['Germany']

    # Initialize the ENTSO-E client
    client = EntsoePandasClient(api_key=api_key)

    # Calculate time ranges
    past_start = target_date - timedelta(days=7)  # 7 days of history
    past_end = target_date - timedelta(hours=1)  # Stop at 23:00 the day before
    future_start = target_date  # Start at 00:00 of target date
    future_end = target_date + timedelta(days=1)  # End at 00:00 next day
    
    # Initialize DataFrames with datetime index
    data_past = pd.DataFrame()
    data_future = pd.DataFrame()
    data_target = pd.DataFrame()

    for country in countries:
        if country == 'Germany':
            area_code = GERMANY_HISTORICAL['new']['code']
        else:
            area_code = ENTSOE_COUNTRY_CODES[country]

        country_short = COUNTRY_DICT[country]

        # Download historical prices for past_data
        prices = download_prices(client, area_code, past_start, past_end)
        
        # Ensure datetime index and hourly frequency
        if prices.index.tz is None:
            prices.index = prices.index.tz_localize(TIMEZONE)
        prices = prices.resample('1h').mean()
        
        # Add prices to past_data
        # Check if prices is a DataFrame and if so, take the first column
        if isinstance(prices,  pd.DataFrame):
            prices = prices.iloc[:, 0]
        data_past[f"{country_short}_price"] = prices
        
        # Download forecast data for future_data
        future_load = download_load(client, area_code, future_start, future_end)
        future_wind_solar = download_wind_solar_forecast(
            client, area_code, country_short, future_start, future_end
        )

        # Convert future_load to a Series if it's a DataFrame
        if isinstance(future_load, pd.DataFrame):
            future_load = future_load.iloc[:, 0]
        
        # Ensure datetime index and hourly frequency for load
        if future_load.index.tz is None:
            future_load.index = future_load.index.tz_localize(TIMEZONE)
        future_load = future_load.resample('1h').mean()
        
        # Add load forecast to future_data
        data_future[f"{country_short}_Load_forecast"] = future_load
        
        # Add wind and solar forecasts to future_data
        if not future_wind_solar.empty:
            if future_wind_solar.index.tz is None:
                future_wind_solar.index = future_wind_solar.index.tz_localize(TIMEZONE)
            future_wind_solar = future_wind_solar.resample('1h').mean()
            for col in future_wind_solar.columns:
                data_future[col] = future_wind_solar[col]

    countries_crossborder = list(CROSSBORDER_COUNTRY_CODES.keys())
    for country_crossborder in countries_crossborder:
        area_code = CROSSBORDER_COUNTRY_CODES[country_crossborder]
        country_short = COUNTRY_DICT[country_crossborder]
        if country_crossborder != 'France':
            area_code_fr = ENTSOE_COUNTRY_CODES['France']
            # Download crossborder flows for past_data
            crossborder_flows_to = download_crossborder_flows(client, area_code_fr, area_code, past_start, past_end)
            data_past[f"FR_{country_short}_flow"] = crossborder_flows_to

            crossborder_flows_from = download_crossborder_flows(client, area_code, area_code_fr, past_start, past_end)
            data_past[f"{country_short}_FR_flow"] = crossborder_flows_from

    
    # Download target data to compare with predictions
    area_code_target = ENTSOE_COUNTRY_CODES['France']
    data_target = download_prices(client, area_code_target, future_start, future_end)
    data_target = data_target.resample('1h').mean()

            
    # Handle missing values and ensure data is properly aligned
    if not data_past.empty:
        # Set frequency to hourly and handle missing values
        data_past = data_past.resample('1h').mean()
        # Use ffill() and bfill() instead of fillna(method=...)
        data_past = data_past.ffill().bfill()
        # Ensure timezone
        if data_past.index.tz is None:
            data_past.index = data_past.index.tz_localize(TIMEZONE)
        elif data_past.index.tz.zone != TIMEZONE:
            data_past.index = data_past.index.tz_convert(TIMEZONE)
    
    if not data_future.empty:
        # Set frequency to hourly and handle missing values
        data_future = data_future.resample('1h').mean()
        # Use ffill() and bfill() instead of fillna(method=...)
        data_future = data_future.ffill().bfill()
        # Ensure timezone
        if data_future.index.tz is None:
            data_future.index = data_future.index.tz_localize(TIMEZONE)
        elif data_future.index.tz.zone != TIMEZONE:
            data_future.index = data_future.index.tz_convert(TIMEZONE)
        # Filter to keep only the target date
        data_future = data_future[data_future.index.date == target_date.date()]

    if not data_target.empty:
        data_target = data_target.resample('1h').mean()
        # Use ffill() and bfill() instead of fillna(method=...)
        data_target = data_target.ffill().bfill()
        # Ensure timezone
        if data_target.index.tz is None:
            data_target.index = data_target.index.tz_localize(TIMEZONE)
        elif data_target.index.tz.zone != TIMEZONE:
            data_target.index = data_target.index.tz_convert(TIMEZONE)
        # Filter to keep only the target date
        data_target = data_target[data_target.index.date == target_date.date()]
    
    return data_past, data_future, data_target


def download_prices(
    client: EntsoePandasClient,
    area_code: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Download day-ahead prices for a specific country and date"""
    try:
        prices = client.query_day_ahead_prices(
            area_code,
            start=start_date,
            end=end_date
        )
        return prices
    except Exception as e:
        raise Exception(f"Error fetching day-ahead prices: {e}")


def download_load(
    client: EntsoePandasClient,
    area_code: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Download load forecast for a specific country and date"""
    try:
        load = client.query_load_forecast(
            area_code,
            start=start_date,
            end=end_date
        )
        return load
    except Exception as e:
        raise Exception(f"Error fetching load forecast: {e}")
    

def download_wind_solar_forecast(
    client: EntsoePandasClient,
    area_code: str,
    country_short: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Download wind and solar forecast for a specific country and date"""
    try:
        wind_solar = client.query_wind_and_solar_forecast(
            area_code,
            start=start_date,
            end=end_date
        )

        # rename columns 
        wind_solar.columns = (
        wind_solar.columns
        .str.replace('Solar', 'Solar_forecast', regex=False)
        .str.replace("Wind Onshore", "Wind_Onshore_forecast", regex=False)
        .str.replace("Wind Offshore", "Wind_Offshore_forecast", regex=False)
    )
        # add country code to columns
        wind_solar.columns = [country_short + "_" + col for col in wind_solar.columns]
        return wind_solar

    except Exception as e:
        raise Exception(f"Error fetching wind and solar forecast: {e}")


def download_crossborder_flows(
    client: EntsoePandasClient,
    country_from: str,
    country_to: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Download crossborder flows data between two countries for a 
    specific date chunk"""
    try:
        flows = client.query_crossborder_flows(
            country_from,
            country_to,
            start=start_date,
            end=end_date
        )
        return flows
    except Exception as e:
        raise Exception(f"Error fetching crossborder flows: {e}")