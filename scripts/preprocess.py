import pandas as pd
import os
import sqlite3
from typing import Optional


def preprocess_entsoe_data(
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
    """
    # Connect to database
    conn = sqlite3.connect(sqlite_path)

    # Read tables
    df_prices = pd.read_sql_query("SELECT * FROM day_ahead_prices", conn)
    df_load = pd.read_sql_query("SELECT * FROM load_data", conn)
    df_generation = pd.read_sql_query("SELECT * FROM generation_data", conn)
    conn.close()

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
    columns_to_drop = ['forecast_load', 'Geothermal', 'Marine', 'Other renewable']
    df_data.drop(columns=columns_to_drop, inplace=True)

    # Map country names to codes
    COUNTRY_DICT = {
        'France': 'FR',
        'Germany': 'DE',
        'Italy': 'IT',
        'Spain': 'ES',
        'Great Britain': 'GB',
        'Netherlands': 'NL',
        'Belgium': 'BE',
        'Switzerland': 'CH',
        'Austria': 'AT',
    }
    df_data['country'] = df_data['country'].map(COUNTRY_DICT)

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
    df_data_pivoted['year'] = df_data_pivoted['datetime'].dt.year
    df_data_pivoted['month'] = df_data_pivoted['datetime'].dt.month
    df_data_pivoted['day'] = df_data_pivoted['datetime'].dt.day
    df_data_pivoted['hour'] = df_data_pivoted['datetime'].dt.hour

    # Save to CSV if output path is provided
    if output_path:
        df_data_pivoted.to_csv(output_path, index=False)

    return df_data_pivoted


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data')
    sqlite_path = os.path.join(DATA_PATH, 'entsoe_data.sqlite')
    output_path = os.path.join(DATA_PATH, 'processed_entsoe_data.csv')
    
    df_processed = preprocess_entsoe_data(
        sqlite_path=sqlite_path,
        output_path=output_path
    )
    print(f"Processed data shape: {df_processed.shape}")

