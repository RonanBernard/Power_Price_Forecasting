# Standard library imports
import sqlite3
import warnings
from pathlib import Path
from typing import Optional, Tuple

# Third-party imports
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Local imports
from scripts.config import (
    DATA_PATH,
    MODELS_PATH,
    COUNTRY_DICT,
    TIMEZONE,
    PREPROCESSING_CONFIG_ATT as PREPROCESSING_CONFIG,
    SECONDS_PER_YEAR,
    SECONDS_PER_DAY
)


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
    """Clean the dataset by removing rows with missing values.
    
    The function processes missing data in three sequential steps:
    1. Wind/Solar Data: Removes rows where wind/solar forecast data is missing,
       except for French offshore wind which is filled with 0s.
    2. Price Data: Removes rows where any price data is missing.
    3. Load Data: Removes rows where load forecast data is missing.

    This order ensures that the most critical data (prices) is handled after
    less critical but still important renewable generation data.

    Args:
        df: DataFrame containing the processed ENTSOE data. Must have columns:
            - Solar/Wind columns with 'Solar' or 'Wind' in name
            - Price columns with 'price' in name
            - Load columns with 'forecast_load' in name
            - Special case: 'FR_Wind Offshore' handled separately

    Returns:
        pd.DataFrame: Cleaned dataset with missing values removed. Prints
        detailed statistics about data removal at each step.

    Raises:
        ValueError: If required columns are missing
        TypeError: If input is not a pandas DataFrame
    """
    initial_rows = len(df)
    df_missing = df.copy()

    # Step 1: Handle missing wind/solar forecast data
    solar_cols = df_missing.columns[
        df_missing.columns.str.contains('Solar')
    ].tolist()
    solar_missing_mask = df_missing[solar_cols].isnull().any(axis=1)
    df_missing = df_missing[~solar_missing_mask]

    wind_cols = df_missing.columns[
        df_missing.columns.str.contains('Wind')
    ].tolist()

    # Fill na with 0 for offshore wind in France, missing data will be removed
    # with solar and onshore wind
    df_missing['FR_Wind_Offshore_forecast'] = df_missing['FR_Wind_Offshore_forecast'].fillna(0)
    wind_cols.remove('FR_Wind_Offshore_forecast')
    wind_missing_mask = df_missing[wind_cols].isnull().any(axis=1)
    df_missing = df_missing[~wind_missing_mask]
    after_wind_solar_rows = len(df_missing)

    # Step 2: Handle missing price data
    price_cols = df_missing.columns[
        df_missing.columns.str.contains('price')
    ].tolist()
    price_missing_mask = df_missing[price_cols].isnull().any(axis=1)
    df_missing = df_missing[~price_missing_mask]
    after_price_rows = len(df_missing)

    # Step 3: Handle missing load data
    load_cols = df_missing.columns[
        df_missing.columns.str.contains('forecast_load')
    ].tolist()
    load_missing_mask = df_missing[load_cols].isnull().any(axis=1)
    df_missing = df_missing[~load_missing_mask]
    after_load_rows = len(df_missing)

    # Step 4: Handle missing flow data
    flow_cols = df_missing.columns[
        df_missing.columns.str.contains('flow')
    ].tolist()
    flow_missing_mask = df_missing[flow_cols].isnull().any(axis=1)
    df_missing = df_missing[~flow_missing_mask]
    final_rows = len(df_missing)

    # Print summary statistics
    print("\nMissing Data Removal Summary:")
    print("-" * 50)
    print(f"Initial number of rows: {initial_rows}")

    print("\nStep 1: Wind/Solar Data")
    print("-" * 25)
    print(f"Total rows removed: {initial_rows - after_wind_solar_rows:,}")
    print(f"Rows remaining: {after_wind_solar_rows:,}")

    print("\nStep 2: Price Data")
    print("-" * 25)
    print(f"Rows removed: {after_wind_solar_rows - after_price_rows:,}")
    print(f"Rows remaining: {after_price_rows:,}")

    print("\nStep 3: Load Data")
    print("-" * 25)
    print(f"Rows removed: {after_price_rows - after_load_rows:,}")
    print(f"Rows remaining: {after_load_rows:,}")

    print("\nStep 4: Flow Data")
    print("-" * 25)
    print(f"Rows removed: {after_load_rows - final_rows:,}")
    print(f"Rows remaining: {final_rows:,}")

    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")

    return df_missing


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
        - Cyclical time encodings (day/year)
        
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
        
        # Yearly features
        year_radians = (
            2 * np.pi * timestamp_seconds / 
            SECONDS_PER_YEAR
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
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
    future_cols += ['Day_sin', 'Day_cos', 'Year_sin', 'Year_cos']

    # Features only known in the past (prices and flows)
    past_cols = [
        col for col in df.columns
        if col not in future_cols + ['datetime']
    ]

    # Save column information
    print("\nSaving column information...")
    columns_info = {
        'past_cols': past_cols,
        'future_cols': future_cols,
        'target_col': target_col
    }
    with open(MODELS_PATH / 'ATT' / 'features_info.json', 'w') as f:
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

    # Initialize lists to store sequences
    past_sequences = []
    future_known_sequences = []
    target_sequences = []
    window_end_times = []  # Store the end time of each window

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
            window_end_times.append(target_start)  # This is midnight of the target day
            
            segment_windows += 1
            total_windows += 1

    # Convert to arrays
    past_sequences = np.array(past_sequences)
    future_known_sequences = np.array(future_known_sequences)
    target_sequences = np.array(target_sequences)

    print("\nSequence shapes:")
    print(f"Past sequences: {past_sequences.shape}")
    print(f"Future known: {future_known_sequences.shape}")
    print(f"Targets: {target_sequences.shape}")

    # Convert window end times to DatetimeIndex
    window_end_times = pd.DatetimeIndex(window_end_times)
    
    print("\nWindow creation summary:")
    print(f"Total windows created: {total_windows}")
    print(f"Windows skipped due to missing data: {skipped_missing_data}")
    print(f"Windows skipped due to shape mismatch: {skipped_shape_mismatch}")
    
    if len(window_end_times) > 1:
        time_deltas = window_end_times[1:] - window_end_times[:-1]
        print("\nWindow spacing:")
        print(f"Mean time between windows: {time_deltas.mean()}")
        print(f"Min time between windows: {time_deltas.min()}")
        print(f"Max time between windows: {time_deltas.max()}")
        
        unique_deltas = sorted(time_deltas.unique())
        print("\nUnique time deltas between windows:")
        for delta in unique_deltas:
            count = (time_deltas == delta).sum()
            print(f"- {delta}: {count:,} occurrences")
    
    return past_sequences, future_known_sequences, target_sequences, window_end_times


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


def main(
    create_model_data: bool = True
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # train
    np.ndarray, np.ndarray, np.ndarray,  # val
    np.ndarray, np.ndarray, np.ndarray   # test
]:
    """
    Main function to process and prepare data for the attention model.

    Args:
        create_model_data: If True, merge ENTSOE data from SQLite database.
                          If False, load existing processed data.

    Returns:
        Tuple containing:
        - X_past_train: Past sequences for training
        - X_future_train: Future known sequences for training
        - y_train: Training targets
        - X_past_test: Past sequences for testing
        - X_future_test: Future known sequences for testing
        - y_test: Test targets
    """
    # Create necessary directories
    data_dir = Path(DATA_PATH)
    data_model_dir = data_dir / 'ATT'
    data_dir.mkdir(exist_ok=True)
    data_model_dir.mkdir(exist_ok=True)

    model_dir = Path(MODELS_PATH) / 'ATT'
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
        df_data = missing_data(df_data)
        df_data = remove_outliers(df_data)
        df_data = merge_fuel_prices(df_data)
        df_data = create_features(df_data)

        # Create sequences
        print("\nCreating sequences...")
        (past_sequences, future_sequences, targets, 
         window_end_times) = create_windows(
            df_data,
            history_hours=PREPROCESSING_CONFIG['HISTORY_HOURS'],
            horizon_hours=PREPROCESSING_CONFIG['HORIZON_HOURS'],
            stride_hours=PREPROCESSING_CONFIG['STRIDE_HOURS']
        )

        # Split chronologically with validation set
        n_samples = len(past_sequences)
        n_test = int(n_samples * PREPROCESSING_CONFIG['TEST_SIZE'])
        n_val = int(n_samples * PREPROCESSING_CONFIG['VAL_SIZE'])
        n_train = n_samples - n_test - n_val

        # Add gap between splits to avoid data leakage
        # Use history_hours as gap to avoid overlap
        gap = PREPROCESSING_CONFIG['HISTORY_HOURS']

        # Training set
        X_past_train = past_sequences[:n_train]
        X_future_train = future_sequences[:n_train]
        y_train = targets[:n_train]
        train_times = window_end_times[:n_train]

        # Validation set (after gap)
        val_start = n_train + gap
        val_end = val_start + n_val
        X_past_val = past_sequences[val_start:val_end]
        X_future_val = future_sequences[val_start:val_end]
        y_val = targets[val_start:val_end]
        val_times = window_end_times[val_start:val_end]

        # Test set (after another gap)
        test_start = val_end + gap
        X_past_test = past_sequences[test_start:]
        X_future_test = future_sequences[test_start:]
        y_test = targets[test_start:]
        test_times = window_end_times[test_start:]

        # Print split information
        print("\nData split summary:")
        print(f"Total sequences: {n_samples:,}")
        print(f"Training sequences: {len(X_past_train):,}")
        print(f"Validation sequences: {len(X_past_val):,}")
        print(f"Test sequences: {len(X_past_test):,}")
        print(f"\nGap between splits: {gap} hours")


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
        pd.to_pickle(train_times, data_model_dir / 'train_times.pkl')
        pd.to_pickle(val_times, data_model_dir / 'val_times.pkl')
        pd.to_pickle(test_times, data_model_dir / 'test_times.pkl')

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
        
        print("Data preprocessing completed successfully!")
        return (
            X_past_train, X_future_train, y_train,
            X_past_val, X_future_val, y_val,
            X_past_test, X_future_test, y_test
        )

    except Exception as e:
        print("Error in preprocessing:", str(e))
        raise


if __name__ == "__main__":
    main(create_model_data=False)
