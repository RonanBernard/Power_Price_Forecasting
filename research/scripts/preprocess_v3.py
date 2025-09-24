# Standard library imports
import sqlite3
import warnings
import os
from pathlib import Path
from typing import Optional, Tuple, List

# Third-party imports
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Local imports
from scripts.config import (
    DATA_PATH,
    MODELS_PATH,
    COUNTRY_DICT,
    TIMEZONE,
    PREPROCESSING_CONFIG_V3 as PREPROCESSING_CONFIG,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    SECONDS_PER_YEAR_LEAP,
    SECONDS_PER_YEAR_NON_LEAP
)

PREPROCESS_VERSION = 'v3'


'''

3rd version of the preprocessing script

Compared to v2
- Focuses on 2015-2020 data.
- Uses all features in the past dataset.

'''

def merge_entsoe_data(
    sqlite_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a dataset by merging ENTSOE prices, load forecast, wind/solar
    forecast, and optionally cross-border flows data.

    The function merges data from multiple ENTSOE tables to create a
    comprehensive dataset for the attention model. Cross-border flows are
    included if the corresponding table exists in the database.

    Args:
        sqlite_path: Path to the SQLite database containing ENTSOE data
        output_path: Optional path to save the processed data as CSV

    Returns:
        pd.DataFrame: Merged dataset with all required features
    """
    # Check if database file exists
    db_path = Path(sqlite_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at: {db_path}\n"
            "Please ensure the database file exists and the path is correct."
        )

    # Print diagnostic information
    print(f"Database path: {db_path}")
    filesize_gb = db_path.stat().st_size / (1024*1024*1024)
    print(f"Database file size: {filesize_gb:.2f} GB")
    print(f"File permissions: {oct(db_path.stat().st_mode)[-3:]}")

    # Connect to database with WAL mode
    try:
        conn = sqlite3.connect(str(db_path), timeout=20)
        # Enable WAL mode
        conn.execute('PRAGMA journal_mode=WAL')
        # Set a larger timeout for busy connections
        conn.execute('PRAGMA busy_timeout=30000')
    except sqlite3.OperationalError as e:
        error_msg = (
            f"Error connecting to database at {db_path}:\n"
            f"{str(e)}\n"
            "Please check file permissions and path validity."
        )
        raise sqlite3.OperationalError(error_msg) from e

    try:
        # Read tables
        print("Reading day_ahead_prices table...")
        query = "SELECT * FROM day_ahead_prices"
        df_prices = pd.read_sql_query(query, conn)
        
        # Standardize German market names to just "Germany"
        df_prices['country'] = df_prices['country'].replace({
            'Germany_Luxembourg': 'Germany',
            'Germany_Austria_Luxembourg': 'Germany'
        })

        print("Reading load_data table...")
        query = "SELECT * FROM load_data"
        df_load = pd.read_sql_query(query, conn)

        print("Reading wind_solar_forecast table...")
        query = "SELECT * FROM wind_solar_forecast"
        df_wind_solar = pd.read_sql_query(query, conn)

        # Try to read cross-border flows if available
        try:
            print("Reading crossborder_flows table...")
            query = "SELECT * FROM crossborder_flows"
            df_flows = pd.read_sql_query(query, conn)
            has_flows = True
        except sqlite3.OperationalError:
            print("Note: crossborder_flows table not found, skipping flow data")
            df_flows = None
            has_flows = False

    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError(
            f"Error reading data from database: {str(e)}\n"
            "Please ensure the database schema is correct."
        ) from e
    finally:
        conn.close()

    print("Processing timestamps...")
    # Convert timestamps to datetime with Paris timezone
    for df in [df_prices, df_load, df_wind_solar, df_flows]:
        df['datetime'] = (
            pd.to_datetime(df['timestamp'], utc=True)
            .dt.tz_convert('Europe/Paris')
        )
        df.drop(columns=['timestamp'], inplace=True)

    # Rename unit columns to be more specific
    df_prices.rename(columns={'unit': 'unit_price'}, inplace=True)
    df_load.rename(columns={'unit': 'unit_load'}, inplace=True)
    df_wind_solar.rename(columns={'unit': 'unit_generation'}, inplace=True)
    df_flows.rename(columns={'unit': 'unit_flow'}, inplace=True)

    df_load.drop(columns=['actual_load'], inplace=True)

    print("Merging datasets...")
    # Merge price and load data
    df_data = pd.merge(
        df_prices,
        df_load,
        on=['datetime', 'country'],
        how='outer'
    )

    # Pivot wind/solar forecast data by type
    df_wind_solar_pivot = df_wind_solar.pivot_table(
        index=['datetime', 'country'],
        columns='forecast_type',
        values='value',
        aggfunc='mean'
    )
    df_wind_solar_pivot.columns.name = None
    df_wind_solar_pivot = df_wind_solar_pivot.reset_index()

    # Merge with wind/solar forecast data
    df_data = pd.merge(
        df_data,
        df_wind_solar_pivot,
        on=['datetime', 'country'],
        how='outer'
    )

    # Process cross-border flows if available
    if has_flows:
        print("Processing cross-border flows...")
        
        # Print the first few rows of df_flows to debug
        print("\nFirst few rows of crossborder flows:")
        print(df_flows[['country_from', 'country_to', 'flow']].head())
        
        # Create a list to store flow data
        flow_data = []
        
        # Process each timestamp
        for ts in df_flows['datetime'].unique():
            ts_data = {'datetime': ts}
            
            # Get flows for this timestamp
            ts_flows = df_flows[df_flows['datetime'] == ts]
            
            # Add each flow as a column
            for _, row in ts_flows.iterrows():
                # Map country names to codes
                from_code = COUNTRY_DICT.get(row['country_from'])
                to_code = COUNTRY_DICT.get(row['country_to'])
                if from_code and to_code:
                    col_name = f"{from_code}_{to_code}_flow"
                    ts_data[col_name] = row['flow']
            
            flow_data.append(ts_data)
        
        # Convert to DataFrame
        df_flows_pivot = pd.DataFrame(flow_data)
        
        # Print column names to debug
        print("\nFlow columns after processing:")
        print(df_flows_pivot.columns.tolist())
        
        # Merge with main dataset
        df_data = pd.merge(
            df_data,
            df_flows_pivot,
            on='datetime',
            how='outer'
        )

    print("Creating final pivoted dataset...")
    # Pivot data by country
    date_columns = ['datetime']
    non_numeric_cols = [
        'country',
        'unit_price',
        'unit_load',
        'unit_generation',
        'unit_flow',
    ]
    
    # Separate flow columns from other columns
    flow_columns = [col for col in df_data.columns if '_flow' in col]
    other_columns = [
        col for col in df_data.columns
        if col not in date_columns + non_numeric_cols + flow_columns
    ]
    
    # First pivot the non-flow data
    df_data_pivoted = df_data.pivot_table(
        index=date_columns,
        columns='country',
        values=other_columns,
        aggfunc='mean'
    )

    # Clean up column names for non-flow data
    df_data_pivoted.columns = [
        f"{COUNTRY_DICT[country]}_{col}" 
        for col, country in df_data_pivoted.columns
    ]
    df_data_pivoted = df_data_pivoted.reset_index()

    # Rename forecast columns to have consistent naming
    df_data_pivoted.columns = (
        df_data_pivoted.columns
        .str.replace("forecast_load", "Load_forecast", regex=False)
        .str.replace('Solar', 'Solar_forecast', regex=False)
        .str.replace("Wind Onshore", "Wind_Onshore_forecast", regex=False)
        .str.replace("Wind Offshore", "Wind_Offshore_forecast", regex=False)
    )
    
    # Add flow columns back
    if flow_columns:
        flow_data = df_data[date_columns + flow_columns]
        df_data_pivoted = pd.merge(
            df_data_pivoted,
            flow_data,
            on='datetime',
            how='outer'
        )

    # Save to CSV if output path is provided
    if output_path:
        # Create output directory if it doesn't exist
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving processed data to: {out_path}")
        df_data_pivoted.to_csv(out_path, index=False)
        print("Save completed!")

    return df_data_pivoted


def missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values.
    
    The function processes missing data in sequential steps:
    1. Wind/Solar Data: Fill missing values with data from 7 days prior,
       then remove remaining missing values. French offshore wind is filled with 0s.
    2. Price Data: Fill missing values with data from 7 days prior,
       then remove remaining missing values.
    3. Load Data: Fill missing values with data from 7 days prior,
       then remove remaining missing values.
    4. Flow Data: Fill missing values with data from 7 days prior,
       then remove remaining missing values.

    This order ensures that the most critical data (prices) is handled after
    less critical but still important renewable generation data.

    Args:
        df: DataFrame containing the processed ENTSOE data. Must have columns:
            - Solar/Wind columns with 'Solar' or 'Wind' in name
            - Price columns with 'price' in name
            - Load columns with 'forecast_load' in name
            - Special case: 'FR_Wind Offshore' handled separately
            - Must have 'datetime' column

    Returns:
        pd.DataFrame: Cleaned dataset with missing values handled. Prints
        detailed statistics about data handling at each step.

    Raises:
        ValueError: If required columns are missing
        TypeError: If input is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if 'datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column")

    initial_rows = len(df)
    df_missing = df.copy()

    # Ensure datetime is properly formatted
    df_missing['datetime'] = (
                pd.to_datetime(df_missing['datetime'], utc=True)
                .dt.tz_convert(TIMEZONE)
            )
    df_missing.set_index('datetime', inplace=True)

    def fill_missing_with_prior_week(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Helper function to fill missing values with data from 7 days prior.
        
        Converts to UTC for consistent datetime arithmetic, then converts back
        to original timezone. This ensures correct handling of DST transitions.
        """
        df_filled = df.copy()
        
        # Convert index to UTC for consistent datetime arithmetic
        orig_tz = df.index.tz
        df_utc = df.tz_convert('UTC')
        df_filled_utc = df_filled.tz_convert('UTC')
        
        # Create a Series mapping each timestamp to its value 7 days prior
        week_offset = pd.Timedelta(days=7)
        prior_week_map = {
            idx: idx - week_offset
            for idx in df_utc.index
        }
        
        # For each column, fill missing values with data from 7 days prior
        for col in columns:
            # Get the values and their indices
            values = df_filled_utc[col]
            missing_mask = values.isnull()
            
            if missing_mask.any():
                # For missing values, look up data from 7 days prior
                missing_dates = values[missing_mask].index
                prior_dates = [prior_week_map[date] for date in missing_dates]
                
                # Get values from 7 days prior where available
                prior_values = df_utc[col].reindex(prior_dates)
                
                # Update only the missing values
                values[missing_mask] = prior_values
                df_filled_utc[col] = values
        
        # Convert back to original timezone
        df_filled = df_filled_utc.tz_convert(TIMEZONE)
        
        # Count how many values were filled
        filled_count = (
            df_filled[columns].notna().sum().sum() - 
            df[columns].notna().sum().sum()
        )
        
        return df_filled, filled_count

    # Step 1: Handle missing wind/solar forecast data
    solar_cols = df_missing.columns[
        df_missing.columns.str.contains('Solar')
    ].tolist()
    
    wind_cols = df_missing.columns[
        df_missing.columns.str.contains('Wind')
    ].tolist()
    
    # Handle missing renewable data
    renewable_cols = solar_cols + wind_cols
    initial_missing_renewable = df_missing[renewable_cols].isnull().sum().sum()
    
    # Process each renewable column separately : take the average of the previous and next value if they are available
    filled_values = 0
    for col in renewable_cols:
        # Create series with values shifted by ±1 hour
        values = df_missing[col]
        values_prev = values.shift(1)  # Shift by 1 row since data is hourly
        values_next = values.shift(-1)  # Shift by -1 row
        
        # Find where we have both previous and next values
        valid_mask = pd.notna(values_prev) & pd.notna(values_next)
        missing_mask = values.isnull()
        can_fill_mask = valid_mask & missing_mask
        
        # Check if timestamps are exactly 1 hour apart
        time_deltas_prev = df_missing.index.to_series().diff()
        time_deltas_next = df_missing.index.to_series().diff(-1)
        hourly_mask = (
            (time_deltas_prev == pd.Timedelta(hours=1)) & 
            (time_deltas_next == -pd.Timedelta(hours=1))
        )
        
        # Only fill where we have adjacent hours and valid values
        final_mask = can_fill_mask & hourly_mask
        
        # Fill missing values with average where possible
        if final_mask.any():
            df_missing.loc[final_mask, col] = (
                values_prev[final_mask] + values_next[final_mask]
            ) / 2
            filled_values += final_mask.sum()
    
    # Count remaining missing values
    remaining_missing_renewable = df_missing[renewable_cols].isnull().sum().sum()
    filled_renewable = filled_values

    # Special handling for French offshore wind
    df_missing['FR_Wind_Offshore_forecast'] = df_missing['FR_Wind_Offshore_forecast'].fillna(0)
    wind_cols.remove('FR_Wind_Offshore_forecast')
    
    # Remove rows where interpolation wasn't possible
    renewable_missing_mask = df_missing[renewable_cols].isnull().any(axis=1)
    df_missing = df_missing[~renewable_missing_mask]
    after_wind_solar_rows = len(df_missing)

    # Step 2: Handle missing price data
    price_cols = df_missing.columns[
        df_missing.columns.str.contains('price')
    ].tolist()
    initial_missing_prices = df_missing[price_cols].isnull().sum().sum()
    df_missing, filled_prices = fill_missing_with_prior_week(df_missing, price_cols)
    remaining_missing_prices = df_missing[price_cols].isnull().sum().sum()
    
    # Remove any remaining missing price data
    price_missing_mask = df_missing[price_cols].isnull().any(axis=1)
    df_missing = df_missing[~price_missing_mask]
    after_price_rows = len(df_missing)

    # Step 3: Handle missing load data
    load_cols = df_missing.columns[
        df_missing.columns.str.contains('forecast_load')
    ].tolist()
    initial_missing_load = df_missing[load_cols].isnull().sum().sum()
    df_missing, filled_load = fill_missing_with_prior_week(df_missing, load_cols)
    remaining_missing_load = df_missing[load_cols].isnull().sum().sum()
    
    # Remove any remaining missing load data
    load_missing_mask = df_missing[load_cols].isnull().any(axis=1)
    df_missing = df_missing[~load_missing_mask]
    after_load_rows = len(df_missing)

    # Step 4: Handle missing flow data
    flow_cols = df_missing.columns[
        df_missing.columns.str.contains('flow')
    ].tolist()
    initial_missing_flow = df_missing[flow_cols].isnull().sum().sum()
    
    # Remove rows with missing flow data
    flow_missing_mask = df_missing[flow_cols].isnull().any(axis=1)
    df_missing = df_missing[~flow_missing_mask]
    final_rows = len(df_missing)

    # Reset index to get datetime back as a column
    df_missing.reset_index(inplace=True)

    # Print summary statistics
    print("\nMissing Data Handling Summary:")
    print("-" * 50)
    print(f"Initial number of rows: {initial_rows:,}")

    print("\nStep 1: Wind/Solar Data")
    print("-" * 25)
    print(f"Initial missing values: {initial_missing_renewable:,}")
    print(f"Values filled by interpolation: {filled_renewable:,}")
    print(f"Remaining missing values: {remaining_missing_renewable:,}")
    print(f"Rows removed: {initial_rows - after_wind_solar_rows:,}")
    print(f"Rows remaining: {after_wind_solar_rows:,}")

    print("\nStep 2: Price Data")
    print("-" * 25)
    print(f"Initial missing values: {initial_missing_prices:,}")
    print(f"Values filled from prior week: {filled_prices:,}")
    print(f"Remaining missing values: {remaining_missing_prices:,}")
    print(f"Rows removed: {after_wind_solar_rows - after_price_rows:,}")
    print(f"Rows remaining: {after_price_rows:,}")

    print("\nStep 3: Load Data")
    print("-" * 25)
    print(f"Initial missing values: {initial_missing_load:,}")
    print(f"Values filled from prior week: {filled_load:,}")
    print(f"Remaining missing values: {remaining_missing_load:,}")
    print(f"Rows removed: {after_price_rows - after_load_rows:,}")
    print(f"Rows remaining: {after_load_rows:,}")

    print("\nStep 4: Flow Data")
    print("-" * 25)
    print(f"Initial missing values: {initial_missing_flow:,}")
    print(f"Rows removed: {after_load_rows - final_rows:,}")
    print(f"Rows remaining: {final_rows:,}")

    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")

    return df_missing

def filter_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """ Keep only data from the years in the list"""

    df_preproc = df.copy()
    df_preproc['datetime'] = (
                pd.to_datetime(df_preproc['datetime'], utc=True)
                .dt.tz_convert(TIMEZONE)
            )
    df_preproc = df_preproc[df_preproc['datetime'].dt.year.isin(years)] # type: ignore
    final_rows = len(df_preproc)
    print(f"Rows removed: {len(df) - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/len(df))*100:.1f}%")
    return df_preproc

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme price values from the dataset.
    
    This function implements two outlier removal strategies:
    1. Individual price threshold: Removes any price above 1000 EUR/MWh
    2. Rolling average threshold: Removes months where 3-month rolling average
       exceeds 100 EUR/MWh
    
    The thresholds are chosen based on typical price patterns in European
    electricity markets. While prices can occasionally exceed these levels
    during extreme events, such cases often indicate market distortions or
    unusual market conditions.

    Args:
        df: DataFrame containing price data. Must have columns containing
           'price' in their names and a 'datetime' column.

    Returns:
        pd.DataFrame: Dataset with outlier rows removed. The function prints
            statistics about the number of rows removed.

    Raises:
        ValueError: If required columns are not found in the DataFrame
        TypeError: If input is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    initial_rows = len(df)
    df_preproc = df.copy()

    df_preproc['datetime'] = (
                pd.to_datetime(df_preproc['datetime'], utc=True)
                .dt.tz_convert(TIMEZONE)
            )

    # Step 1: Remove individual price outliers
    price_cols = df_preproc.columns[
        df_preproc.columns.str.contains('price')
    ].tolist()

    if not price_cols:
        raise ValueError("No price columns found in DataFrame")

    if 'datetime' not in df_preproc.columns:
        raise ValueError("DataFrame must contain 'datetime' column")

    # Remove rows with extreme individual prices
    threshold = PREPROCESSING_CONFIG['PRICE_OUTLIER_THRESHOLD']
    for col in price_cols:
        df_preproc = df_preproc[df_preproc[col] < threshold]

    after_individual_outliers = len(df_preproc)

    # Step 2: Remove months with high rolling averages
    print("\nComputing monthly statistics...")

    monthly_threshold = PREPROCESSING_CONFIG['MONTHLY_PRICE_OUTLIER_THRESHOLD']
    
    monthly_stats = monthly_statistics(df_preproc)
    high_price_months = monthly_stats[
        monthly_stats['3-Month Rolling Average'] > monthly_threshold
    ].index

    # Convert high price months to string format for comparison
    high_price_months_str = high_price_months.strftime('%Y-%m')
    
    # Remove rows from months with high rolling averages
    df_preproc['Month'] = df_preproc['datetime'].dt.strftime('%Y-%m')
    df_preproc = df_preproc[~df_preproc['Month'].isin(high_price_months_str)]
    df_preproc.drop(columns=['Month'], inplace=True)
    
    final_rows = len(df_preproc)

    # Print summary statistics
    print("\nOutlier removal summary:")
    print(f"Step 1 - Individual price threshold ({threshold} EUR/MWh):")
    print(f"Rows removed: {initial_rows - after_individual_outliers:,}")
    print("Step 2 - Rolling average threshold (100 EUR/MWh):")
    print(f"Rows removed: {after_individual_outliers - final_rows:,}")
    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")
    print("\nMonths removed due to high rolling average:")
    for month in sorted(high_price_months_str):
        print(f"- {month}")

    return df_preproc


def merge_fuel_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Merge fuel price data with the main dataset.
    
    Args:
        df: DataFrame containing the main dataset with datetime index
        
    Returns:
        DataFrame with added fuel price columns
        
    Raises:
        FileNotFoundError: If fuel_prices.csv is not found
        ValueError: If required columns are missing or data format is invalid
    """
    fuel_prices_path = DATA_PATH / 'fuel_prices.csv'
    
    try:
        if not fuel_prices_path.exists():
            raise FileNotFoundError(
                f"Fuel prices file not found at: {fuel_prices_path}\n"
                "Please ensure the file exists and the path is correct."
            )
            
        # Read only required columns to save memory
        required_cols = ['Month', 'EUA_EUR', 'TTF_EUR', 'ARA_EUR']
        df_fuel_prices = pd.read_csv(
            fuel_prices_path, 
            usecols=required_cols,
            dtype={
                'Month': str,
                'EUA_EUR': float,
                'TTF_EUR': float,
                'ARA_EUR': float
            }
        )
        
        # Validate fuel prices data
        missing_cols = [
            col for col in required_cols
            if col not in df_fuel_prices.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns in fuel prices data: {missing_cols}"
            )
            
        print("Merging with main dataset...")
        if 'datetime' not in df.columns:
            raise ValueError("Main dataset must contain 'datetime' column")
        
        df['datetime'] = (
                pd.to_datetime(df['datetime'], utc=True)
                .dt.tz_convert(TIMEZONE)
            )

        # Create month column without copying the full dataframe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            month_series = df['datetime'].dt.to_period('M').astype(str)

        # Merge only required columns
        price_cols = ['EUA_EUR', 'TTF_EUR', 'ARA_EUR']
        df_merged = pd.merge(
            df,
            df_fuel_prices[['Month'] + price_cols],
            left_on=month_series,
            right_on='Month',
            how='left'
        )
        
        # Clean up intermediate objects
        del month_series
        del df_fuel_prices
        
        # Drop merge key
        df_merged.drop(columns=['Month'], inplace=True)

        # Report missing values
        missing_prices = df_merged[price_cols].isnull().sum()
        if missing_prices.any():
            print("\nMissing values in price data:")
            for col, count in missing_prices.items():
                if count > 0:
                    print(f"{col}: {count:,} missing values")
                    
        print("Fuel price merging completed!")

        return df_merged
        
    except pd.errors.EmptyDataError:
        raise ValueError("Fuel prices file is empty") from None
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing fuel prices file: {str(e)}") from None
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error merging fuel prices: {str(e)}"
        ) from e


def monthly_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for the DataFrame.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        DataFrame with computed statistics
    """
    if df['datetime'].dtype != 'datetime64[ns]':
        df['datetime'] = (
                    pd.to_datetime(df['datetime'], utc=True)
                    .dt.tz_convert(TIMEZONE)
                )

    # Compute monthly average price
    df['Month'] = df['datetime'].dt.to_period('M')
    monthly_avg_price = df.groupby('Month')['FR_price'].mean()
     
    # Convert Period index to datetime for plotting
    monthly_avg_price.index = (
        monthly_avg_price.index.astype(str).map(pd.Timestamp)
    )

    # Compute 3-month rolling average
    rolling_avg_price = monthly_avg_price.rolling(window=3).mean()

    rolling_avg_price.index = (
        rolling_avg_price.index.astype(str).map(pd.Timestamp)
    )

    # Build final DataFrame with nice column names
    monthly_stats = pd.concat(
        [monthly_avg_price, rolling_avg_price],
        axis=1
    )
    monthly_stats.columns = ['Monthly Average', '3-Month Rolling Average']

    return monthly_stats

def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Create time-based features and lagged price features for the French market.

    Args:
        df: DataFrame with datetime index and price data

    Returns:
        DataFrame with added features:
        - Cyclical time encodings (day/week/year)
        
    Raises:
        ValueError: If input data is invalid or required columns are missing
        TypeError: If input types are incorrect
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    print("Validating input data...")
    if 'datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column")
    
    df_features = df.copy()
        
    print("Adding cyclical time features...")
    try:
        # Convert datetime to timezone-aware index
        df_features['datetime'] = (
            pd.to_datetime(df_features['datetime'], utc=True)
            .dt.tz_convert(TIMEZONE)
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error converting datetime: {str(e)}") from None

    # Sort by datetime to ensure correct sequence and set as index
    df_features = df_features.sort_values('datetime').set_index('datetime')
    
    print("Creating cyclical features...")
    try:
    # Calculate time features efficiently using vectorized operations
        timestamp_seconds = df_features.index.map(pd.Timestamp.timestamp).values
        
        # Daily features
        day_radians = (
            2 * np.pi * timestamp_seconds / 
            SECONDS_PER_DAY
        )
        df_features['Day_sin'] = np.sin(day_radians)
        df_features['Day_cos'] = np.cos(day_radians)

        # Weekly features
        week_radians = (
            2 * np.pi * timestamp_seconds / 
            SECONDS_PER_WEEK
        )
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
        
        # Reset index to get datetime back as a column
        df_features = df_features.reset_index()
        
        return df_features
        
    except Exception as e:
        raise RuntimeError(f"Error creating cyclical features: {str(e)}") from e


def create_windows(
    df: pd.DataFrame,
    history_hours: int = PREPROCESSING_CONFIG['HISTORY_HOURS'],
    horizon_hours: int = PREPROCESSING_CONFIG['HORIZON_HOURS'],
    stride_hours: int = PREPROCESSING_CONFIG['STRIDE_HOURS'],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed sequences for the attention model.

    For each sample ending at time t:
    - past_seq: rows [t-T+1 ... t] over past_cols
      → shape (batch, T, n_past)
    - future_known: rows [t+1 ... t+24] over future_cols
      → shape (batch, 24, n_future)
    - y: FR_price[t+1 ... t+24] (scaled)
      → shape (batch, 24)

    Args:
        df: DataFrame with datetime index and all features
        history_hours: Number of past hours to include (T)
        horizon_hours: Number of future hours to predict (H)
        stride_hours: Hours between consecutive samples

    Returns:
        Tuple containing:
        - past_sequences: Array of shape
          (n_samples, history_hours, n_past_features)
        - future_known: Array of shape
          (n_samples, horizon_hours, n_future_features)
        - targets: Array of shape (n_samples, horizon_hours)
        - past_timestamps: Array of DatetimeIndex objects per sample
        - future_timestamps: Array of DatetimeIndex objects per sample
        - future_window_dates: Array of datetime objects per sample for the 24h future window (midpoint of the 24h window)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if 'datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column")

    # Sort by datetime and remove duplicates
    df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')

    # Separate features
    target_col = 'FR_price'
    
    # Features known in the future (forecasts and calendar)
    future_cols = df.columns[df.columns.str.contains('forecast')].tolist()
    future_cols += ['Day_sin', 'Day_cos', 
                    'Week_sin', 'Week_cos', 
                    'Year_sin', 'Year_cos']

    # Features only known in the past (prices and flows)
    past_cols = [
        col for col in df.columns
        if col not in ['datetime']
    ]

    # Save column information
    print("\nSaving column information...")
    columns_info = {
        'past_cols': past_cols,
        'future_cols': future_cols,
        'target_col': target_col
    }
    with open(MODELS_PATH / PREPROCESS_VERSION / 'features_info.json', 'w') as f:
        json.dump(columns_info, f, indent=4)

    
    print("\nFeature columns:")
    print("\nPast features:")
    for i, col in enumerate(sorted(past_cols), 1):
        print(f"{i:2d}. {col}")
    print(f"\nTotal past features: {len(past_cols)}")
    
    print("\nFuture features:")
    for i, col in enumerate(sorted(future_cols), 1):
        print(f"{i:2d}. {col}")
    print(f"\nTotal future features: {len(future_cols)}")

    # Initialize lists to store sequences and their timestamps
    past_sequences = []
    future_known_sequences = []
    target_sequences = []
    past_timestamps = []  # Store all timestamps for past sequences
    future_timestamps = []  # Store all timestamps for future sequences
    future_window_dates = []  # Store dd/mm/yyyy date string for each future window

    # Get unique timestamps and ensure they're sorted
    timestamps = df.index.sort_values()
    
    # Get the data range
    data_start = timestamps[0]
    data_end = timestamps[-1]
    print(f"\nData range: {data_start} to {data_end}")
    
    # Calculate time deltas between consecutive timestamps
    time_deltas = timestamps[1:] - timestamps[:-1]
    expected_delta = pd.Timedelta(hours=1)
    
    # Print time delta statistics
    print("\nTime delta statistics:")
    print(f"Mean time delta: {time_deltas.mean()}")
    print(f"Min time delta: {time_deltas.min()}")
    print(f"Max time delta: {time_deltas.max()}")
    print(f"Unique time deltas: {sorted(time_deltas.unique())}")
    
    # Find indices where there are large gaps (time delta > 1 hour)
    gap_indices = np.where(time_deltas > expected_delta)[0]
    
    # Split into segments at the large gaps
    segment_starts = np.append([0], gap_indices + 1)
    segment_ends = np.append(gap_indices, len(timestamps) - 1)
    
    print("\nAnalyzing data continuity:")
    print(f"Found {len(gap_indices)} large gaps in the data")
    print(f"Data split into {len(segment_starts)} segments")
    
    # Initialize counters for window statistics
    total_windows = 0
    total_skipped = 0  # Track windows we skip due to data quality
    skipped_shape_mismatch = 0  # Track windows skipped due to shape mismatch
    skipped_missing_data = 0  # Track windows skipped due to missing data
    
    # Process each segment
    for start, end in zip(segment_starts, segment_ends):
        segment_length = end - start + 1
        segment_timestamps = timestamps[start:end+1]
        segment_start_time = segment_timestamps[0]
        segment_end_time = segment_timestamps[-1]
        
        segment_windows = 0
        
        #print(f"\nAnalyzing segment from {segment_start_time} to {segment_end_time}")
        #print(f"Segment length: {segment_length} hours")
        
        # Skip segments that are too short for a full sequence
        if segment_length < history_hours + horizon_hours:
            print(f"""\nSkipping segment from {segment_start_time} to {segment_end_time}:
            too short for sequence ({segment_length} < {history_hours + horizon_hours} hours)""")
            continue
            
        #print(f"Processing segment: {segment_length} continuous hours")
        
        # Get segment timestamps
        segment_start = timestamps[start]
        segment_end = timestamps[end]
        
        # Convert segment boundaries to local time
        local_start = segment_start.tz_convert(TIMEZONE)
        local_end = segment_end.tz_convert(TIMEZONE)
        
        # Find the first midnight that allows a complete history window
        first_possible_start = local_start + pd.Timedelta(hours=history_hours)
        first_midnight = first_possible_start.normalize()
        if first_possible_start != first_midnight:
            # If we're not already at midnight, get the next day's midnight
            first_midnight = first_midnight + pd.Timedelta(days=1)
            
        # Find the last midnight that allows a complete target window
        last_possible_start = local_end - pd.Timedelta(hours=horizon_hours)
        last_midnight = last_possible_start.normalize()
        
        # Skip if we don't have enough continuous data
        if first_midnight > last_midnight:
            print(f"\nSkipping segment: insufficient data for complete windows")
            print(f"First possible window: {first_midnight}")
            print(f"Last possible window: {last_midnight}")
            continue
        
        # Generate daily timestamps at local midnight
        daily_times = pd.date_range(
            start=first_midnight,
            end=last_midnight,
            freq='D',  # 'D' ensures midnight-to-midnight in local time
            tz=TIMEZONE  # Explicitly use local timezone
        )
        
        # For each daily timestamp (midnight), create a window
        for window_start in daily_times:
            # The target window should be the next 24 hours (next day)
            target_start = window_start  # This is midnight
            target_end = target_start + pd.Timedelta(days=1)  # This is midnight next day
            
            # The history window should be the previous week
            history_start = target_start - pd.Timedelta(hours=history_hours)
            
            # Find indices for all our timestamps
            history_start_idx = timestamps.get_indexer([history_start], method='nearest')[0]
            target_start_idx = timestamps.get_indexer([target_start], method='nearest')[0]
            target_end_idx = timestamps.get_indexer([target_end], method='nearest')[0]
            
            # Verify we found the exact timestamps we need
            if (timestamps[target_start_idx] != target_start or 
                timestamps[target_end_idx] != target_end):
                total_skipped += 1
                continue  # Skip if we don't have exact midnight timestamps
            
            # Skip if we don't have enough data before or after
            if history_start_idx < start or target_end_idx > end:
                print(f"\nSkipping window at {target_start}: insufficient data range")
                print(f"Need data from {history_start} to {target_end}")
                print(f"But have data from {timestamps[start]} to {timestamps[end]}")
                skipped_missing_data += 1
                continue
                
            # Verify we have continuous hourly data for both history and target windows
            history_times = timestamps[history_start_idx:target_start_idx]
            target_times = timestamps[target_start_idx:target_end_idx]
            
            # Check history window has hourly data
            history_deltas = history_times[1:] - history_times[:-1]
            if not all(delta == pd.Timedelta(hours=1) for delta in history_deltas):
                continue
                
            # Check target window has hourly data
            target_deltas = target_times[1:] - target_times[:-1]
            if not all(delta == pd.Timedelta(hours=1) for delta in target_deltas):
                continue
            
            # Extract sequences
            past_seq = df.iloc[history_start_idx:target_start_idx][past_cols].values
            future_known = df.iloc[target_start_idx:target_end_idx][future_cols].values
            target = df.iloc[target_start_idx:target_end_idx][target_col].values
            
            # Verify no missing data in the sequence
            if (np.isnan(past_seq).any() or 
                np.isnan(future_known).any() or 
                np.isnan(target).any()):
                skipped_missing_data += 1
                continue
            
            # Check sequence shapes before appending
            expected_past_shape = (history_hours, len(past_cols))
            expected_future_shape = (horizon_hours, len(future_cols))
            expected_target_shape = (horizon_hours,)
            
            if past_seq.shape != expected_past_shape:
                print(f"\nWarning: Unexpected past sequence shape at {target_start}")
                print(f"Expected {expected_past_shape}, got {past_seq.shape}")
                print(f"Past columns: {past_cols}")
                skipped_shape_mismatch += 1
                continue
                
            if future_known.shape != expected_future_shape:
                print(f"\nWarning: Unexpected future sequence shape at {target_start}")
                print(f"Expected {expected_future_shape}, got {future_known.shape}")
                print(f"Future columns: {future_cols}")
                skipped_shape_mismatch += 1
                continue
                
            if target.shape != expected_target_shape:
                print(f"\nWarning: Unexpected target shape at {target_start}")
                print(f"Expected {expected_target_shape}, got {target.shape}")
                skipped_shape_mismatch += 1
                continue
            
            past_sequences.append(past_seq)
            future_known_sequences.append(future_known)
            target_sequences.append(target)
            
            # Store all timestamps for both sequences
            past_timestamps.append(timestamps[history_start_idx:target_start_idx])
            future_timestamps.append(timestamps[target_start_idx:target_end_idx])
            # Store the date (local) for the 24h future window as datetime object for filtering
            future_window_dates.append(target_start + pd.Timedelta(hours=12))
            
            segment_windows += 1
            total_windows += 1

    # Convert to arrays
    past_sequences = np.array(past_sequences)
    future_known_sequences = np.array(future_known_sequences)
    target_sequences = np.array(target_sequences)
    past_timestamps = np.array(past_timestamps, dtype=object)  # Use object dtype for arrays of DatetimeIndex
    future_timestamps = np.array(future_timestamps, dtype=object)
    future_window_dates = np.array(future_window_dates)

    print("\nSequence shapes:")
    print(f"Past sequences: {past_sequences.shape}")
    print(f"Future known: {future_known_sequences.shape}")
    print(f"Targets: {target_sequences.shape}")
    print(f"Past timestamps: {past_timestamps.shape}")
    print(f"Future timestamps: {future_timestamps.shape}")
    print(f"Future window dates: {future_window_dates.shape}")
    
    print("\nWindow creation summary:")
    print(f"Total windows created: {total_windows}")
    print(f"Windows skipped due to missing data: {skipped_missing_data}")
    print(f"Windows skipped due to shape mismatch: {skipped_shape_mismatch}")
    
    if len(past_timestamps) > 1:
        # Calculate time deltas between consecutive windows
        window_starts = pd.DatetimeIndex([ts[0] for ts in past_timestamps])
        time_deltas = window_starts[1:] - window_starts[:-1]
        
        print("\nWindow spacing:")
        print(f"Mean time between windows: {time_deltas.mean()}")
        print(f"Min time between windows: {time_deltas.min()}")
        print(f"Max time between windows: {time_deltas.max()}")
        
        unique_deltas = sorted(time_deltas.unique())
        print("\nUnique time deltas between windows:")
        for delta in unique_deltas:
            count = (time_deltas == delta).sum()
            print(f"- {delta}: {count:,} occurrences")
    
    return (past_sequences, future_known_sequences, target_sequences,
            past_timestamps, future_timestamps, future_window_dates)


def create_pipeline(
    X_past: np.ndarray,
    X_future: np.ndarray,
    save_path: Optional[Path] = None
) -> Tuple[Pipeline, Pipeline]:
    """
    Create and fit preprocessing pipelines for past and future features.

    Args:
        X_past: Training data for past features
        X_future: Training data for future features
        save_path: Optional path to save the fitted pipelines

    Returns:
        Tuple containing:
        - past_pipeline: Pipeline for past features
        - future_pipeline: Pipeline for future features
    """
    # Create pipelines
    past_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
    ])

    future_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
    ])

    # Reshape 3D sequences to 2D for sklearn pipeline
    n_samples_past, seq_len_past, n_features_past = X_past.shape
    n_samples_future, seq_len_future, n_features_future = X_future.shape

    X_past_2d = X_past.reshape(-1, n_features_past)
    X_future_2d = X_future.reshape(-1, n_features_future)

    # Fit pipelines
    print("Fitting preprocessing pipelines...")
    past_pipeline.fit(X_past_2d)
    future_pipeline.fit(X_future_2d)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        past_path = save_path.parent / 'past_pipeline.joblib'
        future_path = save_path.parent / 'future_pipeline.joblib'
        print(f"Saving pipelines to: {save_path.parent}")
        joblib.dump(past_pipeline, past_path)
        joblib.dump(future_pipeline, future_path)

    return past_pipeline, future_pipeline


def transform_sequences(
    past_sequences: np.ndarray,
    future_sequences: np.ndarray,
    past_pipeline: Pipeline,
    future_pipeline: Pipeline
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform sequences using fitted preprocessing pipelines.

    Args:
        past_sequences: Past feature sequences
        future_sequences: Future known sequences
        past_pipeline: Fitted pipeline for past features
        future_pipeline: Fitted pipeline for future features

    Returns:
        Tuple containing transformed sequences
    """
    # Get original sequence dimensions
    n_samples_past, seq_len_past, _ = past_sequences.shape
    n_samples_future, seq_len_future, _ = future_sequences.shape

    # Reshape to 2D for sklearn pipeline
    past_2d = past_sequences.reshape(n_samples_past * seq_len_past, -1)
    future_2d = future_sequences.reshape(n_samples_future * seq_len_future, -1)

    # Transform
    past_transformed = past_pipeline.transform(past_2d)
    future_transformed = future_pipeline.transform(future_2d)

    # Get transformed feature dimensions
    n_features_past = past_transformed.shape[1]
    n_features_future = future_transformed.shape[1]

    # Reshape back to 3D
    past_transformed = past_transformed.reshape(
        n_samples_past, seq_len_past, n_features_past
    )
    future_transformed = future_transformed.reshape(
        n_samples_future, seq_len_future, n_features_future
    )

    return past_transformed, future_transformed

def merge_data(
    X_past_train: np.ndarray,
    X_future_train: np.ndarray,
    y_train: np.ndarray,
    train_past_times: np.ndarray,
    train_future_times: np.ndarray,
):
    
    '''
    Merge all data into one list and return it. Easier to manipulate one variable.
    
    '''

    data = [X_past_train, X_future_train, y_train, train_past_times, train_future_times]
    return data

def split_data(
    data: list,
):
    '''
    Split the list into the different sets.
    '''

    X_past_train = data[0]
    X_future_train = data[1]
    y_train = data[2]
    train_past_times = data[3]
    train_future_times = data[4]
    
    return X_past_train, X_future_train, y_train, train_past_times, train_future_times


def shuffle_data(data_train, data_val, data_test):

    """
    Merge all sequences and shuffle them together, then split back into train/val/test sets.
    Maintains correspondence between sequences, targets, and timestamps.
    Completely breaks chronological order across all sets.
    
    Args:
        All input arrays for train, validation, and test sets
        
    Returns:
        All arrays shuffled and redistributed into the original set sizes
    """

    X_past_train, X_future_train, y_train, train_past_times, train_future_times = split_data(data_train)
    X_past_val, X_future_val, y_val, val_past_times, val_future_times = split_data(data_val)
    X_past_test, X_future_test, y_test, test_past_times, test_future_times = split_data(data_test)

    # Get original set sizes
    n_train = len(X_past_train)
    n_val = len(X_past_val)
    n_test = len(X_past_test)
    
    # Concatenate all sequences

    print(f"X_past_train.shape: {X_past_train.shape}, X_past_val.shape: {X_past_val.shape}, X_past_test.shape: {X_past_test.shape}")

    X_past_all = np.concatenate([X_past_train, X_past_val, X_past_test])
    X_future_all = np.concatenate([X_future_train, X_future_val, X_future_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    past_times_all = np.concatenate([train_past_times, val_past_times, test_past_times])
    future_times_all = np.concatenate([train_future_times, val_future_times, test_future_times])
    
    # Generate random permutation for all data
    total_samples = len(X_past_all)
    perm = np.random.permutation(total_samples)
    
    # Shuffle all arrays with the same permutation
    X_past_shuffled = X_past_all[perm]
    X_future_shuffled = X_future_all[perm]
    y_shuffled = y_all[perm]
    past_times_shuffled = past_times_all[perm]
    future_times_shuffled = future_times_all[perm]
    
    # Split back into train/val/test sets
    # Training set
    X_past_train_new = X_past_shuffled[:n_train]
    X_future_train_new = X_future_shuffled[:n_train]
    y_train_new = y_shuffled[:n_train]
    train_past_times_new = past_times_shuffled[:n_train]
    train_future_times_new = future_times_shuffled[:n_train]
    
    # Validation set
    X_past_val_new = X_past_shuffled[n_train:n_train + n_val]
    X_future_val_new = X_future_shuffled[n_train:n_train + n_val]
    y_val_new = y_shuffled[n_train:n_train + n_val]
    val_past_times_new = past_times_shuffled[n_train:n_train + n_val]
    val_future_times_new = future_times_shuffled[n_train:n_train + n_val]
    
    # Test set
    X_past_test_new = X_past_shuffled[n_train + n_val:]
    X_future_test_new = X_future_shuffled[n_train + n_val:]
    y_test_new = y_shuffled[n_train + n_val:]
    test_past_times_new = past_times_shuffled[n_train + n_val:]
    test_future_times_new = future_times_shuffled[n_train + n_val:]
    
    print("\nData shuffling summary:")
    print(f"Total samples shuffled: {total_samples:,}")
    print(f"New training set size: {len(X_past_train_new):,}")
    print(f"New validation set size: {len(X_past_val_new):,}")
    print(f"New test set size: {len(X_past_test_new):,}")

    data_train_new = merge_data(X_past_train_new, X_future_train_new, y_train_new, train_past_times_new, train_future_times_new)
    data_val_new = merge_data(X_past_val_new, X_future_val_new, y_val_new, val_past_times_new, val_future_times_new)
    data_test_new = merge_data(X_past_test_new, X_future_test_new, y_test_new, test_past_times_new, test_future_times_new)
    
    return data_train_new, data_val_new, data_test_new

def data_split_classic(
    past_sequences: np.ndarray,
    future_sequences: np.ndarray,
    targets: np.ndarray,
    past_times: np.ndarray,
    future_times: np.ndarray,
    future_window_dates: np.ndarray,
    data_model_dir: Path,
    model_dir: Path
):
    
    data_model_dir = data_model_dir / "classic"
    data_model_dir.mkdir(exist_ok=True)

    model_dir = model_dir / "classic"
    model_dir.mkdir(exist_ok=True)

    # Calculate initial split sizes
    n_samples = len(past_sequences)
    n_test = int(n_samples * PREPROCESSING_CONFIG['TEST_SIZE'])
    n_val = int(n_samples * PREPROCESSING_CONFIG['VAL_SIZE'])
    n_train = n_samples - n_test - n_val

    # Calculate minimum required gap in hours to avoid data leakage
    min_gap_hours = PREPROCESSING_CONFIG['HISTORY_HOURS'] + PREPROCESSING_CONFIG['HORIZON_HOURS']
    
    # Training set (initial split)
    train_end = n_train
    X_past_train = past_sequences[:train_end]
    X_future_train = future_sequences[:train_end]
    y_train = targets[:train_end]

    
    # Find valid start of validation set by checking temporal gaps
    val_start = train_end
    last_train_time = future_times[train_end - 1][-1]  # Last timestamp of last training sequence
    while val_start < len(future_times):  # Check if val start is within the future times, in case it is moving too far in the future
        current_time = past_times[val_start][0]  # First timestamp of current sequence
        gap_hours = (current_time - last_train_time).total_seconds() / 3600
        if gap_hours >= min_gap_hours:
            break
        val_start += 1
    
    # Validation set
    val_end = val_start + n_val
    X_past_val = past_sequences[val_start:val_end]
    X_future_val = future_sequences[val_start:val_end]
    y_val = targets[val_start:val_end]
    
    # Find valid start of test set
    test_start = val_end
    last_val_time = future_times[val_end - 1][-1] if val_end > 0 else last_train_time  # Last timestamp of last validation sequence
    while test_start < len(future_times):
        current_time = past_times[test_start][0]  # First timestamp of current sequence
        gap_hours = (current_time - last_val_time).total_seconds() / 3600
        if gap_hours >= min_gap_hours:
            break
        test_start += 1
    
    # Test set
    X_past_test = past_sequences[test_start:]
    X_future_test = future_sequences[test_start:]
    y_test = targets[test_start:]

    
    # Print detailed split information
    print("\nTemporal split verification:")
    print(f"Last training sequence ends at:   {future_times[train_end - 1][-1]}")
    print(f"First validation sequence starts: {past_times[val_start][0] if val_start < len(past_times) else 'N/A'}")
    if val_start < len(past_times):
        train_val_gap = (past_times[val_start][0] - future_times[train_end - 1][-1]).total_seconds() / 3600
        print(f"Gap between train-val: {train_val_gap:.1f} hours")
    
    if val_end > 0 and test_start < len(past_times):
        print(f"Last validation sequence ends at: {future_times[val_end - 1][-1]}")
        print(f"First test sequence starts:      {past_times[test_start][0]}")
        val_test_gap = (past_times[test_start][0] - future_times[val_end - 1][-1]).total_seconds() / 3600
        print(f"Gap between val-test: {val_test_gap:.1f} hours")

    # Print split information
    print("\nData split summary:")
    print(f"Total sequences: {n_samples:,}")
    print(f"Training sequences: {len(X_past_train):,}")
    print(f"Validation sequences: {len(X_past_val):,}")
    print(f"Test sequences: {len(X_past_test):,}")
    print(f"\nGap between splits: {min_gap_hours} hours")


    # Save data arrays
    np.save(data_model_dir / 'X_past_train.npy', X_past_train)
    np.save(data_model_dir / 'X_future_train.npy', X_future_train)
    np.save(data_model_dir / 'y_train.npy', y_train)
    np.save(data_model_dir / 'X_past_val.npy', X_past_val)
    np.save(data_model_dir / 'X_future_val.npy', X_future_val)
    np.save(data_model_dir / 'y_val.npy', y_val)
    np.save(data_model_dir / 'X_past_test.npy', X_past_test)
    np.save(data_model_dir / 'X_future_test.npy', X_future_test)
    np.save(data_model_dir / 'y_test.npy', y_test)
    
    # Save datetime indices
                # Save timestamps using pickle to preserve DatetimeIndex objects
    pd.to_pickle(past_times[:train_end], data_model_dir / 'train_past_times.pkl')
    pd.to_pickle(past_times[val_start:val_end], data_model_dir / 'val_past_times.pkl')
    pd.to_pickle(past_times[test_start:], data_model_dir / 'test_past_times.pkl')

    # Save future timestamps using pickle
    pd.to_pickle(future_times[:train_end], data_model_dir / 'train_future_times.pkl')
    pd.to_pickle(future_times[val_start:val_end], data_model_dir / 'val_future_times.pkl')
    pd.to_pickle(future_times[test_start:], data_model_dir / 'test_future_times.pkl')

    # Save future window dates using pickle
    pd.to_pickle(future_window_dates[:train_end], data_model_dir / 'train_future_window_dates.pkl')
    pd.to_pickle(future_window_dates[val_start:val_end], data_model_dir / 'val_future_window_dates.pkl')
    pd.to_pickle(future_window_dates[test_start:], data_model_dir / 'test_future_window_dates.pkl')

    # Create and fit preprocessing pipelines
    print("\nPreprocessing sequences...")
    past_pipeline, future_pipeline = create_pipeline(
        X_past_train,
        X_future_train,
        save_path=model_dir / 'pipelines.joblib'
    )

    # Transform sequences
    X_past_train_transformed, X_future_train_transformed = transform_sequences(
        X_past_train, X_future_train,
        past_pipeline, future_pipeline
    )
    X_past_val_transformed, X_future_val_transformed = transform_sequences(
        X_past_val, X_future_val,
        past_pipeline, future_pipeline
    )
    X_past_test_transformed, X_future_test_transformed = transform_sequences(
        X_past_test, X_future_test,
        past_pipeline, future_pipeline
    )

    # Save processed data
    print("\nSaving processed sequences...")
    # Save transformed sequences
    np.save(
        data_model_dir / 'X_past_train_transformed.npy', 
        X_past_train_transformed
    )
    np.save(
        data_model_dir / 'X_future_train_transformed.npy',
        X_future_train_transformed
    )
    np.save(
        data_model_dir / 'X_past_val_transformed.npy',
        X_past_val_transformed
    )
    np.save(
        data_model_dir / 'X_future_val_transformed.npy',
        X_future_val_transformed
    )
    np.save(
        data_model_dir / 'X_past_test_transformed.npy',
        X_past_test_transformed
    )
    np.save(
        data_model_dir / 'X_future_test_transformed.npy',
        X_future_test_transformed
    )
    
    print("Data split classic completed successfully!")
    return None

def data_split_rolling_horizon(
    past_sequences: np.ndarray,
    future_sequences: np.ndarray,
    targets: np.ndarray,
    past_times: np.ndarray,
    future_times: np.ndarray,
    future_window_dates: np.ndarray,
    data_model_dir: Path,
    model_dir: Path
):
    
    data_model_dir = data_model_dir / "rolling_horizon"
    data_model_dir.mkdir(exist_ok=True)

    model_dir = model_dir / "rolling_horizon"
    model_dir.mkdir(exist_ok=True)
    
    cv = PREPROCESSING_CONFIG['CV']
    rolling_horizon_val_size = PREPROCESSING_CONFIG['ROLLING_HORIZON_VAL_SIZE']

    # Calculate minimum required gap in hours to avoid data leakage
    min_gap_hours = PREPROCESSING_CONFIG['HISTORY_HOURS'] + PREPROCESSING_CONFIG['HORIZON_HOURS']

    # Calculate initial split sizes
    n_samples = len(past_sequences)
    n_test = int(n_samples * PREPROCESSING_CONFIG['TEST_SIZE'])
    n_train = n_samples - n_test
    
    # Training set (initial split)
    train_end = n_train
    X_past_train = past_sequences[:train_end]
    X_future_train = future_sequences[:train_end]
    y_train = targets[:train_end]

    # Find valid start of test set by checking temporal gaps
    test_start = train_end
    last_train_time = future_times[train_end - 1][-1]  # Last timestamp of last training sequence
    while test_start < len(future_times):
        current_time = past_times[test_start][0]  # First timestamp of current sequence
        gap_hours = (current_time - last_train_time).total_seconds() / 3600
        if gap_hours >= min_gap_hours:
            break
        test_start += 1
    
    # Validation set
    test_end = test_start + n_test
    X_past_test = past_sequences[test_start:test_end]
    X_future_test = future_sequences[test_start:test_end]
    y_test = targets[test_start:test_end]

    np.save(data_model_dir / 'X_past_test.npy', X_past_test)
    np.save(data_model_dir / 'X_future_test.npy', X_future_test)
    np.save(data_model_dir / 'y_test.npy', y_test)

    pd.to_pickle(past_times[test_start:], data_model_dir / 'test_past_times.pkl')
    pd.to_pickle(future_times[test_start:], data_model_dir / 'test_future_times.pkl')
    pd.to_pickle(future_window_dates[test_start:], data_model_dir / 'test_future_window_dates.pkl')

    for i in range(cv):

        # Select only part of train data for first rolling horizon
        n_samples_cv = (i + 1) * int(n_train / cv)
        X_past_cv = X_past_train[:n_samples_cv]
        X_future_cv = X_future_train[:n_samples_cv]
        y_cv = y_train[:n_samples_cv]

        
        # Proceed normally with the rest of the data
        n_val_cv = int(n_samples_cv * rolling_horizon_val_size)
        n_train_cv = n_samples_cv - n_val_cv

        train_end_cv = n_train_cv
        X_past_train_cv = X_past_cv[:train_end_cv]
        X_future_train_cv = X_future_cv[:train_end_cv]
        y_train_cv = y_cv[:train_end_cv]


        # Find valid start of test set by checking temporal gaps
        val_start_cv = train_end_cv
        last_train_time_cv = future_times[train_end_cv - 1][-1]  # Last timestamp of last training sequence
        while val_start_cv < len(X_future_cv):
            current_time = past_times[val_start_cv][0]  # First timestamp of current sequence
            gap_hours = (current_time - last_train_time_cv).total_seconds() / 3600
            if gap_hours >= min_gap_hours:
                break
            val_start_cv += 1

        # Validation set
        val_end_cv = val_start_cv + n_val_cv
        X_past_val_cv = X_past_cv[val_start_cv:]
        X_future_val_cv = X_future_cv[val_start_cv:]
        y_val_cv = y_cv[val_start_cv:]

        # Print progress information
        print(f"\nProcessing fold {i+1}/{cv}")
        print(f"Training samples: {len(X_past_train_cv)}")
        print(f"Validation samples: {len(X_past_val_cv)}")
        print(f"Training period: {past_times[0][0]} to {future_times[train_end_cv-1][-1]}")
        print(f"Validation period: {past_times[val_start_cv][0]} to {future_times[val_end_cv-1][-1]}")

        # Save data arrays
        np.save(data_model_dir / f'X_past_train_fold_{i+1}.npy', X_past_train_cv)
        np.save(data_model_dir / f'X_future_train_fold_{i+1}.npy', X_future_train_cv)
        np.save(data_model_dir / f'y_train_fold_{i+1}.npy', y_train_cv)
        np.save(data_model_dir / f'X_past_val_fold_{i+1}.npy', X_past_val_cv)
        np.save(data_model_dir / f'X_future_val_fold_{i+1}.npy', X_future_val_cv)
        np.save(data_model_dir / f'y_val_fold_{i+1}.npy', y_val_cv)


        # Save datetime indices
        # Save timestamps using pickle to preserve DatetimeIndex objects
        pd.to_pickle(past_times[:train_end_cv], data_model_dir / f'train_past_times_fold_{i+1}.pkl')
        pd.to_pickle(past_times[val_start_cv:val_end_cv], data_model_dir / f'val_past_times_fold_{i+1}.pkl')

        # Save future timestamps using pickle
        pd.to_pickle(future_times[:train_end_cv], data_model_dir / f'train_future_times_fold_{i+1}.pkl')
        pd.to_pickle(future_times[val_start_cv:val_end_cv], data_model_dir / f'val_future_times_fold_{i+1}.pkl')

        # Save future window dates using pickle
        pd.to_pickle(future_window_dates[:train_end_cv], data_model_dir / f'train_future_window_dates_fold_{i+1}.pkl')
        pd.to_pickle(future_window_dates[val_start_cv:val_end_cv], data_model_dir / f'val_future_window_dates_fold_{i+1}.pkl')

        # Create and fit preprocessing pipelines
        print("\nPreprocessing sequences...")
        past_pipeline_cv, future_pipeline_cv = create_pipeline(
            X_past_train_cv,
            X_future_train_cv,
            save_path=model_dir / f'pipelines_cv{i}.joblib'
        )

        # Transform sequences
        X_past_train_transformed_cv, X_future_train_transformed_cv = transform_sequences(
            X_past_train_cv, X_future_train_cv,
            past_pipeline_cv, future_pipeline_cv
        )
        X_past_val_transformed_cv, X_future_val_transformed_cv = transform_sequences(
            X_past_val_cv, X_future_val_cv,
            past_pipeline_cv, future_pipeline_cv
        )

        if i + 1 == cv:
            X_past_test_transformed, X_future_test_transformed = transform_sequences(
                X_past_test, X_future_test,
                past_pipeline_cv, future_pipeline_cv
            )

        # Save processed data
        print("\nSaving processed sequences...")
        # Save transformed sequences
        np.save(
            data_model_dir / f'X_past_train_transformed_fold_{i+1}.npy', 
            X_past_train_transformed_cv
        )
        np.save(
            data_model_dir / f'X_future_train_transformed_fold_{i+1}.npy',
            X_future_train_transformed_cv
        )
        np.save(
            data_model_dir / f'X_past_val_transformed_fold_{i+1}.npy',
            X_past_val_transformed_cv
        )
        np.save(
            data_model_dir / f'X_future_val_transformed_fold_{i+1}.npy',
            X_future_val_transformed_cv
        )

    np.save(
        data_model_dir / 'X_past_test_transformed.npy',
        X_past_test_transformed
    )
    np.save(
        data_model_dir / 'X_future_test_transformed.npy',
        X_future_test_transformed
    )

    print("Data split rolling horizon completed successfully!")
    return None


def data_full_set(
    past_sequences: np.ndarray,
    future_sequences: np.ndarray,
    targets: np.ndarray,
    past_times: np.ndarray,
    future_times: np.ndarray,
    future_window_dates: np.ndarray,
    data_model_dir: Path,
    model_dir: Path
):
    
    data_model_dir = data_model_dir / "full_set"
    data_model_dir.mkdir(exist_ok=True)

    model_dir = model_dir / "full_set"
    model_dir.mkdir(exist_ok=True)

    X_past = past_sequences
    X_future = future_sequences
    y = targets

    print("\nPreprocessing sequences...")
    past_pipeline, future_pipeline = create_pipeline(
        X_past,
        X_future,
        save_path=model_dir / 'pipelines.joblib'
    )

    X_past_transformed, X_future_transformed = transform_sequences(
        X_past, X_future,
        past_pipeline, future_pipeline
    )

    np.save(data_model_dir / 'X_past.npy', X_past)
    np.save(data_model_dir / 'X_future.npy', X_future)
    np.save(data_model_dir / 'X_past_transformed.npy', X_past_transformed)
    np.save(data_model_dir / 'X_future_transformed.npy', X_future_transformed)
    np.save(data_model_dir / 'y.npy', y)

    pd.to_pickle(past_times, data_model_dir / 'past_times.pkl')
    pd.to_pickle(future_times, data_model_dir / 'future_times.pkl')
    pd.to_pickle(future_window_dates, data_model_dir / 'future_window_dates.pkl')

    print("Data full set completed successfully!")
    return None


def main(
    create_model_data: bool = True,
    data_split_type: str = 'classic'
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,  # train
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,  # val
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray   # test
]:
    """
    Main function to process and prepare data for the attention model.

    Args:
        create_model_data: If True, merge ENTSOE data from SQLite database.
                          If False, load existing processed data.

    Returns:
        Tuple containing:
        Training data:
        - X_past_train: Past sequences for training
        - X_future_train: Future known sequences for training
        - y_train: Training targets
        - train_past_times: Timestamps for past sequences in training
        - train_future_times: Timestamps for future sequences in training
        
        Validation data:
        - X_past_val: Past sequences for validation
        - X_future_val: Future known sequences for validation
        - y_val: Validation targets
        - val_past_times: Timestamps for past sequences in validation
        - val_future_times: Timestamps for future sequences in validation
        
        Test data:
        - X_past_test: Past sequences for testing
        - X_future_test: Future known sequences for testing
        - y_test: Test targets
        - test_past_times: Timestamps for past sequences in testing
        - test_future_times: Timestamps for future sequences in testing
    """
    # Create necessary directories
    data_dir = Path(DATA_PATH)
    data_model_dir = data_dir / PREPROCESS_VERSION
    data_dir.mkdir(exist_ok=True)
    data_model_dir.mkdir(exist_ok=True)

    model_dir = Path(MODELS_PATH) / PREPROCESS_VERSION
    model_dir.mkdir(exist_ok=True)

    # Define paths
    sqlite_path = data_dir / 'entsoe_data.sqlite'
    output_path = data_model_dir / 'model_data.csv'

    try:
        if create_model_data:
            print("Merging ENTSOE data...")
            df_data = merge_entsoe_data(
                sqlite_path=str(sqlite_path),
                output_path=str(output_path)
            )
            print(f"Dataset shape after merge: {df_data.shape}")
        else:
            print(f"Loading existing processed data from {output_path}")
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Processed data file not found at: {output_path}\n"
                    "Please set create_model_data=True to create it."
                )
            df_data = pd.read_csv(output_path)
            print(f"Dataset shape: {df_data.shape}")

        # Clean data
        df_data = filter_years(df_data, [2015, 2016, 2017, 2018, 2019, 2020])
        df_data = missing_data(df_data)
        df_data = remove_outliers(df_data)
        df_data = merge_fuel_prices(df_data)
        df_data = create_features(df_data)

        # Create sequences
        print("\nCreating sequences...")
        (past_sequences, future_sequences, targets,
         past_times, future_times, future_window_dates) = create_windows(
            df_data,
            history_hours=PREPROCESSING_CONFIG['HISTORY_HOURS'],
            horizon_hours=PREPROCESSING_CONFIG['HORIZON_HOURS'],
            stride_hours=PREPROCESSING_CONFIG['STRIDE_HOURS']
        )

        if data_split_type == 'classic':
            data_split_classic(
                past_sequences,
                future_sequences,
                targets,
                past_times,
                future_times,
                future_window_dates,
                data_model_dir,
                model_dir
            )
        
        elif data_split_type == 'rolling_horizon':
            data_split_rolling_horizon(
                past_sequences,
                future_sequences,
                targets,
                past_times,
                future_times,
                future_window_dates,
                data_model_dir,
                model_dir
            )

        data_full_set(
            past_sequences,
            future_sequences,
            targets,
            past_times,
            future_times,
            future_window_dates,
            data_model_dir,
            model_dir
        )
        
        print("Data preprocessing completed successfully!")
        
        return None

    except Exception as e:
        print("Error in preprocessing:", str(e))
        raise


if __name__ == "__main__":
    main(create_model_data=False, data_split_type='classic')
