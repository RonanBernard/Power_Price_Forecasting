"""Service for data preprocessing using saved sklearn pipelines."""

import os
import json
from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd
from scripts.config import (
    DEFAULT_FUEL_PRICES,
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    ENTSOE_API_KEY,
    API_MODELS_PATH
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
        
        # Yearly features
        year_radians = 2 * np.pi * timestamp_seconds / SECONDS_PER_YEAR
        df_features['Year_sin'] = np.sin(year_radians)
        df_features['Year_cos'] = np.cos(year_radians)
        
        # Clean up
        del timestamp_seconds, day_radians, year_radians
    
        return df_features
        
    except Exception as e:
        raise RuntimeError(f"Error creating cyclical features: {str(e)}") from e


def main(target_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function for preprocessing data.
    
    Args:
        target_date: The target date for data preprocessing
        
    Returns:
        Tuple containing:
        - data_past: DataFrame with historical data and fuel prices
        - data_future: DataFrame with future data and cyclical features
    """
    api_key = ENTSOE_API_KEY
    data_past, data_future, data_target = download_entsoe_data(
        api_key, 
        target_date=target_date
    )

    data_past = add_default_fuel_prices(data_past)
    data_future = create_features(data_future)

    # Align the columns order as in features_info.json
    features_path = os.path.join(API_MODELS_PATH, 'features_info.json')
    with open(features_path) as f:
        features_info = json.load(f)
    
    data_past = data_past[features_info['past_cols']]
    data_future = data_future[features_info['future_cols']]

    return data_past, data_future, data_target
