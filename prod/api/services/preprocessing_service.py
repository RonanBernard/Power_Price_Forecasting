"""Service for data preprocessing using saved sklearn pipelines."""

import os
import json
from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd
import joblib
from api.config import (
    DEFAULT_FUEL_PRICES,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    SECONDS_PER_YEAR_LEAP,
    SECONDS_PER_YEAR_NON_LEAP,
    API_MODELS_PATH,
    PREPROCESSING_CONFIG_ATT
)


from api.services.entsoe_service import download_entsoe_data


def add_default_fuel_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Merge fuel prices into the DataFrame."""
    df = df.copy()

    ARA_EUR = DEFAULT_FUEL_PRICES['ARA_USD'] * DEFAULT_FUEL_PRICES['USD_EUR']

    # Create constant Series aligned with df.index
    df.loc[:, 'TTF_EUR'] = pd.Series(
        DEFAULT_FUEL_PRICES["TTF_EUR"], 
        index=df.index
    )
    df.loc[:, 'EUA_EUR'] = pd.Series(
        DEFAULT_FUEL_PRICES["EUA_EUR"], 
        index=df.index
    )
    df.loc[:, 'ARA_EUR'] = pd.Series(ARA_EUR, index=df.index)

    return df


def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Create time-based cyclical features for time series data.

    Args:
        df: DataFrame with datetime index and price data

    Returns:
        DataFrame with added features:
        - Day_sin: Daily cyclical pattern (sine)
        - Day_cos: Daily cyclical pattern (cosine)
        - Year_sin: Yearly cyclical pattern (sine)
        - Year_cos: Yearly cyclical pattern (cosine)
        
    Raises:
        ValueError: If input data is invalid or required columns are missing
        TypeError: If input types are incorrect
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    df_features = df.copy()
    
    print("Creating cyclical features...")
    try:
        # Calculate time features efficiently using vectorized operations
        timestamp_seconds = (
            df_features.index.map(pd.Timestamp.timestamp).values
        )
        
        # Daily features
        day_radians = 2 * np.pi * timestamp_seconds / SECONDS_PER_DAY
        df_features['Day_sin'] = np.sin(day_radians)
        df_features['Day_cos'] = np.cos(day_radians)

        # Weekly features
        week_radians = (2 * np.pi * timestamp_seconds / SECONDS_PER_WEEK)
        df_features['Week_sin'] = np.sin(week_radians)
        df_features['Week_cos'] = np.cos(week_radians)
        
        # Yearly features
        # Handle leap years and non-leap years correctly
        # Get year and check if it's a leap year using pandas functionality
        is_leap_year = pd.DatetimeIndex(df_features.index).is_leap_year
        
        # Create array of seconds per year based on leap year status
        seconds_per_year = np.where(
            is_leap_year,
            SECONDS_PER_YEAR_LEAP,
            SECONDS_PER_YEAR_NON_LEAP
        )
            
        year_radians = (
            2 * np.pi * timestamp_seconds / 
            seconds_per_year
        )
        df_features['Year_sin'] = np.sin(year_radians)
        df_features['Year_cos'] = np.cos(year_radians)
        
        # Clean up
        del timestamp_seconds, day_radians, year_radians
    
        return df_features
        
    except Exception as e:
        raise RuntimeError(f"Error creating cyclical features: {str(e)}") from e


def main(target_date: datetime, api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function for preprocessing data.
    
    Args:
        target_date: The target date for data preprocessing
        api_key: ENTSOE API key for data fetching
        
    Returns:
        Tuple containing:
        - data_past: DataFrame with historical data and fuel prices
        - data_future: DataFrame with future data and cyclical features
    """
    data_past, data_future, data_target = download_entsoe_data(
        api_key, 
        target_date=target_date
    )

    # Ensure we have exactly HISTORY_HOURS of past data
    if len(data_past) > PREPROCESSING_CONFIG_ATT['HISTORY_HOURS']:
        data_past = data_past.iloc[-PREPROCESSING_CONFIG_ATT['HISTORY_HOURS']:]
    elif len(data_past) < PREPROCESSING_CONFIG_ATT['HISTORY_HOURS']:
        raise ValueError(
            f"Not enough historical data. Expected {PREPROCESSING_CONFIG_ATT['HISTORY_HOURS']} "
            f"hours but got {len(data_past)} hours."
        )

    # Ensure we have exactly HORIZON_HOURS of future data
    if len(data_future) > PREPROCESSING_CONFIG_ATT['HORIZON_HOURS']:
        data_future = data_future.iloc[:PREPROCESSING_CONFIG_ATT['HORIZON_HOURS']]
    elif len(data_future) < PREPROCESSING_CONFIG_ATT['HORIZON_HOURS']:
        raise ValueError(
            f"Not enough future data. Expected {PREPROCESSING_CONFIG_ATT['HORIZON_HOURS']} "
            f"hours but got {len(data_future)} hours."
        )

    data_past = add_default_fuel_prices(data_past)
    data_past = create_features(data_past)
    data_future = create_features(data_future)

    # Align the columns order as in features_info.json
    features_path = os.path.join(API_MODELS_PATH, 'features_info.json')
    with open(features_path) as f:
        features_info = json.load(f)
    
    data_past = data_past[features_info['past_cols']]
    data_future = data_future[features_info['future_cols']]

    past_pipeline = joblib.load(os.path.join(API_MODELS_PATH, "past_pipeline.joblib"))
    future_pipeline = joblib.load(os.path.join(API_MODELS_PATH, "future_pipeline.joblib"))

    data_past_transformed = past_pipeline.transform(data_past)
    data_future_transformed = future_pipeline.transform(data_future)

    return data_past_transformed, data_future_transformed, data_target