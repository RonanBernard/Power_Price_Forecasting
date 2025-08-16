# Standard library imports
import sqlite3
import warnings
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Local imports
from scripts.config import (
    DATA_PATH,
    MODELS_PATH,
    COUNTRY_DICT,
    TIMEZONE,
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    PREPROCESSING_CONFIG_V1 as PREPROCESSING_CONFIG
)

def merge_entsoe_data(
    sqlite_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a dataset by merging ENTSOE prices, load forecast, and
    wind/solar forecast data.

    Args:
        sqlite_path: Path to the SQLite database containing ENTSOE data
        output_path: Optional path to save the processed data as CSV

    Returns:
        pd.DataFrame
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

        print("Reading load_data table...")
        query = "SELECT * FROM load_data"
        df_load = pd.read_sql_query(query, conn)

        print("Reading wind_solar_forecast table...")
        query = "SELECT * FROM wind_solar_forecast"
        df_wind_solar = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError(
            f"Error reading data from database: {str(e)}\n"
            "Please ensure the database schema is correct."
        ) from e
    finally:
        conn.close()

    print("Processing timestamps...")
    # Convert timestamps to datetime with Paris timezone
    for df in [df_prices, df_load, df_wind_solar]:
        df['datetime'] = (
            pd.to_datetime(df['timestamp'], utc=True)
            .dt.tz_convert('Europe/Paris')
        )
        df.drop(columns=['timestamp'], inplace=True)

    # Rename unit columns to be more specific
    df_prices.rename(columns={'unit': 'unit_price'}, inplace=True)
    df_load.rename(columns={'unit': 'unit_load'}, inplace=True)
    df_wind_solar.rename(columns={'unit': 'unit_generation'}, inplace=True)

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

    # Map country names to codes
    df_data['country'] = df_data['country'].map(COUNTRY_DICT)

    print("Creating final pivoted dataset...")
    # Pivot data by country
    date_columns = ['datetime']
    non_numeric_cols = [
        'country',
        'unit_price',
        'unit_load',
        'unit_generation',
    ]
    pivot_columns = [
        col for col in df_data.columns
        if col not in date_columns + non_numeric_cols
    ]

    df_data_pivoted = df_data.pivot_table(
        index=date_columns,
        columns='country',
        values=pivot_columns,
        aggfunc='mean'
    )

    # Clean up column names
    df_data_pivoted.columns = [
        f"{country}_{col}" for col, country in df_data_pivoted.columns
    ]
    df_data_pivoted = df_data_pivoted.reset_index()

    # Add time-based features
    # df_data_pivoted['Year'] = df_data_pivoted['datetime'].dt.year
    # df_data_pivoted['Month'] = df_data_pivoted['datetime'].dt.month
    # df_data_pivoted['Day'] = df_data_pivoted['datetime'].dt.day
    # df_data_pivoted['Hour'] = df_data_pivoted['datetime'].dt.hour

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
    df_missing['FR_Wind Offshore'] = df_missing['FR_Wind Offshore'].fillna(0)
    wind_cols.remove('FR_Wind Offshore')
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
    print(f"Rows removed: {after_price_rows - final_rows:,}")
    print(f"Final number of rows: {final_rows:,}")

    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")

    return df_missing


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme price values from the dataset.
    
    This function implements two outlier removal strategies:
    1. Individual price threshold: Removes any price above 1000 EUR/MWh
    2. Rolling average threshold: Removes months where 3-month rolling average > 100 EUR/MWh
    
    The thresholds are chosen based on typical price patterns in European electricity
    markets. While prices can occasionally exceed these levels during extreme events,
    such cases often indicate market distortions or unusual market conditions.

    Args:
        df: DataFrame containing price data. Must have columns containing 'price'
           in their names and a 'datetime' column.

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
    print(f"\nOutlier removal summary:")
    print(f"Step 1 - Individual price threshold ({threshold} EUR/MWh):")
    print(f"Rows removed: {initial_rows - after_individual_outliers:,}")
    print(f"Step 2 - Rolling average threshold (100 EUR/MWh):")
    print(f"Rows removed: {after_individual_outliers - final_rows:,}")
    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")
    print(f"\nMonths removed due to high rolling average:")
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
            dtype={'Month': str, 'EUA_EUR': float, 'TTF_EUR': float, 'ARA_EUR': float}
        )
        
        # Validate fuel prices data
        missing_cols = [col for col in required_cols if col not in df_fuel_prices.columns]
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
        raise RuntimeError(f"Unexpected error merging fuel prices: {str(e)}") from e


def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Create time-based features and lagged price features for the French market.

    Args:
        df: DataFrame with datetime index and price data
        lag_hours: Number of hours to lag the price data (default: 72 hours/3 days)

    Returns:
        DataFrame with added features:
        - Cyclical time encodings (day/year)
        - Lagged price features
        
    Raises:
        ValueError: If input data is invalid or required columns are missing
        TypeError: If input types are incorrect
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Use default lag hours from config if not provided

    lag_hours = PREPROCESSING_CONFIG['LAG_HOURS']
    if not isinstance(lag_hours, int) or lag_hours <= 0:
        raise ValueError("lag_hours must be a positive integer")

    print("Validating input data...")
    if 'datetime' not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column")
        
    # Verify we have French price data
    fr_price_cols = [col for col in df.columns if 'FR' in col and 'price' in col]
    if not fr_price_cols:
        raise ValueError("No French price data found in DataFrame")
        
    print("Filtering French data...")
    try:
        # Filter French columns without copying the full dataframe
        fr_cols = ['datetime'] + [
            col for col in df.columns 
            if 'FR' in col or col in ['Day_sin', 'Day_cos', 'Year_sin', 'Year_cos', 'EUA_EUR', 'TTF_EUR', 'ARA_EUR']
        ]
        df_features = df[fr_cols].copy()

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

        except Exception as e:
            raise RuntimeError(f"Error creating cyclical features: {str(e)}") from e

        print("Adding lags...")
        try:
            # Get the French price column
            price_col = fr_price_cols[0]
            price_series = df_features[price_col]

            # Pre-allocate lag columns for better memory efficiency
            lag_cols = [
                f'price_lag_{h}h' 
                for h in range(24, 24 + lag_hours + 1)
            ]
            
            for h in range(24, 24 + lag_hours + 1):
                df_features[f'price_lag_{h}h'] = price_series.shift(
                    freq=pd.Timedelta(hours=h)
                )

            # Drop rows with missing lags
            df_features = df_features.dropna(subset=lag_cols)
            if len(df_features) == 0:
                raise ValueError("No data remaining after removing missing values")

            print(f"Final dataset shape: {df_features.shape}")
            # Reset index to get datetime back as a column
            df_features = df_features.reset_index()
            return df_features

        except Exception as e:
            raise RuntimeError(f"Error creating lagged features: {str(e)}") from e

    except Exception as e:
        raise RuntimeError(f"Error in feature creation: {str(e)}") from e
    
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
    monthly_avg_price.index = monthly_avg_price.index.astype(str).map(pd.Timestamp)

    # Compute 3-month rolling average
    rolling_avg_price = monthly_avg_price.rolling(window=3).mean()

    rolling_avg_price.index = rolling_avg_price.index.astype(str).map(pd.Timestamp)

    # Build final DataFrame with nice column names
    monthly_stats = pd.concat(
        [monthly_avg_price, rolling_avg_price],
        axis=1
    )
    monthly_stats.columns = ['Monthly Average', '3-Month Rolling Average']

    return monthly_stats


def create_pipeline(
    X_train: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Pipeline:
    """Create and fit a preprocessing pipeline for feature scaling.
    
    This function creates a scikit-learn pipeline that:
    1. Imputes missing values with zeros
    2. Standardizes features using StandardScaler
    3. Preserves cyclical features (sin/cos) without scaling
    
    The pipeline is fit on the training data and can be used to transform
    both training and test data consistently.

    Args:
        X_train: Training data to fit the pipeline. Must contain all features
                that will be used for model training.
        save_path: Optional path to save the fitted pipeline. If None, the
                pipeline is saved to the default models directory.

    Returns:
        Pipeline: Fitted scikit-learn pipeline ready for transforming data.
        
    Raises:
        ValueError: If X_train is empty or contains invalid data types
        TypeError: If X_train is not a pandas DataFrame
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
        
    if X_train.empty:
        raise ValueError("X_train cannot be empty")

    # Create the basic pipeline with imputer and scaler
    pipe = Pipeline(
        [
            ('imputer_zero', SimpleImputer(fill_value=0)),
            ('stdscaler', StandardScaler()),
        ]
    ).set_output(transform='pandas')

    # Exclude cyclical features and datetime from scaling
    excluded = ['Day_sin', 'Day_cos', 'Year_sin', 'Year_cos', 'datetime']
    selected_columns = [col for col in X_train.columns if col not in excluded]

    # Create the full pipeline with column transformer
    preproc_pipeline = ColumnTransformer(
        [('base_pipeline', pipe, selected_columns)],
        remainder='passthrough',
    ).set_output(transform='pandas')

    print("\nFitting preprocessing pipeline...")
    preproc_pipeline.fit(X_train)

    # Save the pipeline
    if save_path is None:
        save_path = Path(MODELS_PATH) / 'V1' / 'preproc_pipeline.joblib'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving pipeline to: {save_path}")
    joblib.dump(preproc_pipeline, save_path)

    return preproc_pipeline

def main(
    create_model_data: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Main function to process and prepare data for model training.
    
    This function handles the complete data preprocessing pipeline:
    1. Loads/merges raw data
    2. Cleans data (missing values, outliers)
    3. Creates features
    4. Splits into train/test sets
    5. Scales features
    
    The function carefully prevents data leakage by:
    - Splitting data before feature creation
    - Fitting preprocessing pipeline only on training data
    - Applying transformations separately to train/test sets
    
    Args:
        create_model_data: If True, merge ENTSOE data from SQLite database.
                          If False, load existing processed data.
                          
    Returns:
        tuple containing:
        - X_train_transformed: Transformed training features
        - X_test_transformed: Transformed test features
        - y_train: Training target values
        - y_test: Test target values
    """
    # Create necessary directories
    data_dir = Path(DATA_PATH)
    model_dir = data_dir / 'V1'
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Define paths
    sqlite_path = data_dir / 'entsoe_data.sqlite'
    output_path = model_dir / 'model_data.csv'

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
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

    print("\nCleaning data...")
    # These steps are safe to do before splitting as they don't learn from the data
    df_clean = missing_data(df_data)
    df_clean = remove_outliers(df_clean)
    df_clean = merge_fuel_prices(df_clean)

    print("\nSplitting into train and test sets chronologically...")
    # Sort by datetime to ensure chronological order
    df_clean = df_clean.sort_values('datetime')
    
    # Calculate split point based on test size
    n_samples = len(df_clean)
    n_test = int(n_samples * PREPROCESSING_CONFIG['TEST_SIZE'])
    n_val = int(n_samples * PREPROCESSING_CONFIG['VAL_SIZE'])
    n_train = n_samples - n_test - n_val

    # Avoid data leakage by having a gap = lag_hours between each split
    lag_hours = 24 + PREPROCESSING_CONFIG['LAG_HOURS'] + 1
    
    # Split the data chronologically
    df_train = df_clean.iloc[:n_train].copy()
    df_val = df_clean.iloc[n_train+lag_hours:n_train+n_val].copy()
    df_test = df_clean.iloc[n_train+n_val+lag_hours:].copy()
    
    # Print split information
    train_start = df_train['datetime'].min()
    train_end = df_train['datetime'].max()
    val_start = df_val['datetime'].min()
    val_end = df_val['datetime'].max()
    test_start = df_test['datetime'].min()
    test_end = df_test['datetime'].max()
    
    print(f"\nTrain period: {train_start} to {train_end}")
    print(f"Validation period: {val_start} to {val_end}")
    print(f"Test period: {test_start} to {test_end}")
    print(f"Train samples: {len(df_train):,}")
    print(f"Validation samples: {len(df_val):,}")
    print(f"Test samples: {len(df_test):,}")

    print("\nCreating features...")
    # Create features separately for train and test to prevent leakage
    df_train_features = create_features(df_train)
    df_val_features = create_features(df_val)
    df_test_features = create_features(df_test)

    print("\nSplitting into features and target...")
    X_train = df_train_features.drop(columns=['FR_price'])
    y_train = df_train_features['FR_price']
    X_val = df_val_features.drop(columns=['FR_price'])
    y_val = df_val_features['FR_price']
    X_test = df_test_features.drop(columns=['FR_price'])
    y_test = df_test_features['FR_price']

    print("\nCreating and fitting preprocessing pipeline...")
    # Create and fit preprocessing pipeline only on training data
    pipeline = create_pipeline(X_train)

    print("\nTransforming features...")
    # Transform both train and test data
    X_train_transformed = pipeline.transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    X_train_transformed.drop(columns=['remainder__datetime'], inplace=True)
    X_val_transformed.drop(columns=['remainder__datetime'], inplace=True)
    X_test_transformed.drop(columns=['remainder__datetime'], inplace=True)

    print("\nSaving processed datasets...")
    # Save train/test splits
    X_train.to_csv(model_dir / 'X_train.csv', index=False)
    X_train_transformed.to_csv(model_dir / 'X_train_transformed.csv', index=False)
    y_train.to_csv(model_dir / 'y_train.csv', index=False)
    X_val.to_csv(model_dir / 'X_val.csv', index=False)
    X_val_transformed.to_csv(model_dir / 'X_val_transformed.csv', index=False)
    y_val.to_csv(model_dir / 'y_val.csv', index=False)
    X_test.to_csv(model_dir / 'X_test.csv', index=False)
    X_test_transformed.to_csv(model_dir / 'X_test_transformed.csv', index=False)
    y_test.to_csv(model_dir / 'y_test.csv', index=False)
    
    print("Data preprocessing completed successfully!")
    return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    main(create_model_data=False)
