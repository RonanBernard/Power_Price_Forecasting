import pandas as pd
import numpy as np
import os
import sqlite3
from typing import Optional
from scripts.config import (
    DATA_PATH,
    COUNTRY_DICT,
    FUEL_PRICES_FILES
)

def merge_entsoe_data_V0(
    sqlite_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess power data from ENTSOE database by:
    1. Merging price, load and generation data
    2. Converting timestamps to Europe/Paris timezone
    3. Pivoting generation types by country
    4. Adding time-based features

    Args:
        sqlite_path: Path to the SQLite database containing ENTSOE data
        output_path: Optional path to save the processed data as CSV

    Returns:
        pd.DataFrame: Processed dataframe ready for modeling

    Raises:
        FileNotFoundError: If the SQLite database file doesn't exist
        sqlite3.OperationalError: If there's an error accessing the database
    """
    # Check if database file exists
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(
            f"SQLite database not found at: {sqlite_path}\n"
            "Please ensure the database file exists and the path is correct."
        )

    # Print diagnostic information
    print(f"Database path: {sqlite_path}")
    filesize_gb = os.path.getsize(sqlite_path) / (1024*1024*1024)
    print(f"Database file size: {filesize_gb:.2f} GB")
    print(f"File permissions: {oct(os.stat(sqlite_path).st_mode)[-3:]}")

    # Connect to database with WAL mode
    try:
        conn = sqlite3.connect(sqlite_path, timeout=20)
        # Enable WAL mode
        conn.execute('PRAGMA journal_mode=WAL')
        # Set a larger timeout for busy connections
        conn.execute('PRAGMA busy_timeout=30000')
    except sqlite3.OperationalError as e:
        error_msg = (
            f"Error connecting to database at {sqlite_path}:\n"
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

        print("Reading generation_data table...")
        query = "SELECT * FROM generation_data"
        df_generation = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError(
            f"Error reading data from database: {str(e)}\n"
            "Please ensure the database schema is correct."
        ) from e
    finally:
        conn.close()

    print("Processing timestamps...")
    # Convert timestamps to datetime with Paris timezone
    for df in [df_prices, df_load, df_generation]:
        df['datetime'] = (
            pd.to_datetime(df['timestamp'], utc=True)
            .dt.tz_convert('Europe/Paris')
        )
        df.drop(columns=['timestamp'], inplace=True)

    # Rename unit columns to be more specific
    df_prices.rename(columns={'unit': 'unit_price'}, inplace=True)
    df_load.rename(columns={'unit': 'unit_load'}, inplace=True)
    df_generation.rename(columns={'unit': 'unit_generation'}, inplace=True)

    print("Merging datasets...")
    # Merge price and load data
    df_data = pd.merge(
        df_prices,
        df_load,
        on=['datetime', 'country'],
        how='outer'
    )

    # Pivot generation data by type
    df_gen_pivot = df_generation.pivot_table(
        index=['datetime', 'country'],
        columns='generation_type',
        values='value',
        aggfunc='mean'
    )
    df_gen_pivot.columns.name = None
    df_gen_pivot = df_gen_pivot.reset_index()

    # Merge with generation data
    df_data = pd.merge(
        df_data,
        df_gen_pivot,
        on=['datetime', 'country'],
        how='outer'
    )

    # Drop less relevant columns
    columns_to_drop = [
        'forecast_load', 'Geothermal', 'Marine', 'Other renewable'
    ]
    df_data.drop(columns=columns_to_drop, inplace=True)

    # Map country names to codes
    df_data['country'] = df_data['country'].map(COUNTRY_DICT)

    print("Creating final pivoted dataset...")
    # Pivot data by country
    date_columns = ['datetime']
    non_numeric_cols = ['country', 'unit_price', 'unit_load']
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
    '''
    df_data_pivoted['Year'] = df_data_pivoted['datetime'].dt.year
    df_data_pivoted['Month'] = df_data_pivoted['datetime'].dt.month
    df_data_pivoted['Day'] = df_data_pivoted['datetime'].dt.day
    df_data_pivoted['Hour'] = df_data_pivoted['datetime'].dt.hour
    '''

    # Save to CSV if output path is provided
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving processed data to: {output_path}")
        df_data_pivoted.to_csv(output_path, index=False)
        print("Save completed!")

    return df_data_pivoted



def missing_data_V0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing rows with missing values in three steps:
    1. Remove rows where all generation data is missing for a country
    2. Remove rows with missing price data
    3. Remove rows with missing load data

    Args:
        df: DataFrame containing the processed ENTSOE data

    Returns:
        pd.DataFrame: Cleaned dataset with missing values removed
    """
    initial_rows = len(df)
    df_missing = df.copy()

    # Step 1: Handle missing generation data
    country_missing_stats = {}
    missing_index_gen = []

    for country in list(set(COUNTRY_DICT.values())):
        # Get all columns for the country
        country_cols = df_missing.columns[
            df_missing.columns.str.contains(country)
        ].tolist()

        if not country_cols:
            continue

        # Get generation columns (exclude price and load)
        gen_cols = [
            col for col in country_cols
            if not (col.endswith('_price') or col.endswith('_actual_load'))
        ]

        if not gen_cols:
            continue

        # Find rows where all generation data is missing
        missing_mask = df_missing[gen_cols].isnull().all(axis=1)
        missing_indices = df_missing[missing_mask].index.tolist()

        country_missing_stats[country] = len(missing_indices)
        missing_index_gen.extend(missing_indices)

    # Remove rows with all missing generation data
    df_missing = df_missing.drop(missing_index_gen)
    after_gen_rows = len(df_missing)

    # Step 2: Handle missing price data
    price_cols = df_missing.columns[
        df_missing.columns.str.contains('price')
    ].tolist()
    price_missing_mask = df_missing[price_cols].isnull().any(axis=1)
    df_missing = df_missing[~price_missing_mask]
    after_price_rows = len(df_missing)

    # Step 3: Handle missing load data
    load_cols = df_missing.columns[
        df_missing.columns.str.contains('actual_load')
    ].tolist()
    load_missing_mask = df_missing[load_cols].isnull().any(axis=1)
    df_missing = df_missing[~load_missing_mask]
    final_rows = len(df_missing)

    # Print summary statistics
    print("\nMissing Data Removal Summary:")
    print("-" * 50)
    print(f"Initial number of rows: {initial_rows}")

    print("\nStep 1: Generation Data")
    print("-" * 25)
    for country, count in country_missing_stats.items():
        if count > 0:
            print(f"{country}: {count:,} rows removed")
    print(f"Rows after generation cleaning: {after_gen_rows:,}")
    print(f"Total rows removed: {initial_rows - after_gen_rows:,}")

    print("\nStep 2: Price Data")
    print("-" * 25)
    print(f"Rows removed: {after_gen_rows - after_price_rows:,}")
    print(f"Rows remaining: {after_price_rows:,}")

    print("\nStep 3: Load Data")
    print("-" * 25)
    print(f"Rows removed: {after_price_rows - final_rows:,}")
    print(f"Final number of rows: {final_rows:,}")

    print(f"\nTotal rows removed: {initial_rows - final_rows:,}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.1f}%")

    return df_missing




if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)

    sqlite_path = os.path.join(DATA_PATH, 'entsoe_data.sqlite')
    output_path = os.path.join(DATA_PATH, 'model_data.csv')

    try:
        df_data = merge_entsoe_data_V1(
            sqlite_path=sqlite_path,
            output_path=output_path
        )
        print(f"Dataset shape: {df_data.shape}")
    except (FileNotFoundError, sqlite3.OperationalError) as e:
        print(f"Error: {str(e)}")
        exit(1)
