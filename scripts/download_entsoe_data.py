from entsoe import EntsoePandasClient
import pandas as pd
import os
import time
import sqlite3
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
from .config import (
    DATA_PATH,
    TIMEZONE,
    ENTSOE_COUNTRY_CODES,
    GERMANY_HISTORICAL,
    ENTSOE_DATA_TYPES,
    ENTSOE_GENERATION_TYPES,
    ENTSOE_CHUNK_SIZE,
    DB_FILE
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database lock for thread safety
DB_LOCK = threading.Lock()


def get_db_connection() -> sqlite3.Connection:
    """Create a new database connection with timeout and better concurrency"""
    conn = sqlite3.connect(
        DB_FILE,
        timeout=60.0,  # Wait up to 60 seconds for lock
        isolation_level='IMMEDIATE'  # Get immediate transaction lock
    )
    # Use Write-Ahead Logging for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    """Initialize SQLite database with required tables"""
    os.makedirs(DATA_PATH, exist_ok=True)
    
    with DB_LOCK:  # Use global lock for initialization
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Create table for day-ahead prices
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS day_ahead_prices (
                timestamp DATETIME,
                country TEXT,
                price FLOAT,
                unit TEXT DEFAULT 'EUR/MWh',
                PRIMARY KEY (timestamp, country)
            )
            ''')
            
            # Create table for load data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS load_data (
                timestamp DATETIME,
                country TEXT,
                actual_load FLOAT,
                forecast_load FLOAT,
                unit TEXT DEFAULT 'MW',
                PRIMARY KEY (timestamp, country)
            )
            ''')
            
            # Create table for generation data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_data (
                timestamp DATETIME,
                country TEXT,
                generation_type TEXT,
                value FLOAT,
                unit TEXT DEFAULT 'MW',
                PRIMARY KEY (timestamp, country, generation_type)
            )
            ''')
            
            # Create table to track downloaded chunks
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_log (
                country TEXT,
                data_type TEXT,
                start_date DATETIME,
                end_date DATETIME,
                download_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                PRIMARY KEY (country, data_type, start_date, end_date)
            )
            ''')
            
            # Create indices for better query performance
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prices_time_country
            ON day_ahead_prices(timestamp, country)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_load_time_country
            ON load_data(timestamp, country)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_gen_time_country_type
            ON generation_data(timestamp, country, generation_type)
            ''')
            
            conn.commit()


def is_chunk_downloaded(
    conn: sqlite3.Connection,
    country: str,
    data_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> bool:
    """Check if a specific chunk has already been downloaded successfully"""
    with DB_LOCK:  # Use lock for checking download status
        cursor = conn.cursor()
        cursor.execute('''
        SELECT status FROM download_log
        WHERE country = ? AND data_type = ? 
        AND start_date = ? AND end_date = ?
        AND status = 'success'
        ''', (
            country,
            data_type,
            start_date.isoformat(),
            end_date.isoformat()
        ))
        
        return cursor.fetchone() is not None


def create_date_chunks(
    start_year: int,
    end_year: int
) -> List[Dict[str, pd.Timestamp]]:
    """Create a list of date ranges in 3-month chunks"""
    chunks = []
    start_date = pd.Timestamp(f"{start_year}-01-01", tz=TIMEZONE)
    end_date = pd.Timestamp(f"{end_year}-12-31 23:59:59", tz=TIMEZONE)
    
    current_date = start_date
    while current_date < end_date:
        chunk_end = min(
            current_date + pd.Timedelta(days=ENTSOE_CHUNK_SIZE-1),
            end_date
        )
        chunks.append({
            'start': current_date,
            'end': chunk_end
        })
        current_date = chunk_end + pd.Timedelta(days=1)
    
    return chunks


def get_german_config(date: pd.Timestamp) -> Tuple[str, str]:
    """Get the appropriate German market configuration for a given date"""
    if date <= GERMANY_HISTORICAL['old']['end_date']:
        return (
            GERMANY_HISTORICAL['old']['name'],
            GERMANY_HISTORICAL['old']['code']
        )
    else:
        return (
            GERMANY_HISTORICAL['new']['name'],
            GERMANY_HISTORICAL['new']['code']
        )


def process_generation_data(
    df: pd.DataFrame,
    country_name: str
) -> pd.DataFrame:
    """Process generation data into a format suitable for the database"""
    # Ensure the index is timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize(TIMEZONE)
    
    # Resample to hourly frequency
    df = df.resample('h').mean()
    
    processed_data = []
    timestamps = df.index
    
    # Log the available columns for debugging
    logging.info(
        f"Available generation types for {country_name}: {df.columns.tolist()}"
    )
    
    for column in df.columns:
        # Handle tuple columns (type, aggregation)
        if isinstance(column, tuple):
            gen_type = column[0]
            agg_type = column[1]
            
            # Special handling for Hydro Pumped Storage
            if gen_type == 'Hydro Pumped Storage':
                if agg_type == 'Actual Aggregated':
                    gen_type = 'Hydro Pumped Storage_Generation'
                elif agg_type == 'Actual Consumption':
                    gen_type = 'Hydro Pumped Storage_Consumption'
                else:
                    continue
            # For other types, only take Actual Aggregated
            elif agg_type != 'Actual Aggregated':
                continue
        else:
            gen_type = column.split(' [')[0] if ' [' in column else column
            gen_type = gen_type.strip()
        
        # Skip if not in our list of generation types
        if gen_type not in ENTSOE_GENERATION_TYPES:
            logging.info(
                f"Skipping generation type for {country_name}: {gen_type}"
            )
            continue
        
        # Get the values for this generation type
        values = df[column].values
        
        # Add each timestamp-value pair to the processed data
        for ts, val in zip(timestamps, values):
            if pd.notna(val):  # Only add non-null values
                processed_data.append({
                    'timestamp': ts,
                    'country': country_name,
                    'generation_type': gen_type,
                    'value': val
                })
    
    # Log the number of records being saved
    if processed_data:
        logging.info(
            f"Processing {len(processed_data)} records for {country_name}"
        )
    
    return pd.DataFrame(processed_data)


def save_to_database(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    country: str,
    date_chunk: Dict[str, pd.Timestamp],
    data_type: str
) -> bool:
    """Save data to database with proper locking"""
    with DB_LOCK:
        try:
            # Save to database
            df.to_sql(table, conn, if_exists='append', index=False)
            
            # Log successful download
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO download_log 
            (country, data_type, start_date, end_date, status)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                country,
                data_type,
                date_chunk['start'].isoformat(),
                date_chunk['end'].isoformat(),
                'success'
            ))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error saving to database: {str(e)}")
            return False


def download_prices(
    client: EntsoePandasClient,
    country_name: str,
    area_code: str,
    date_chunk: Dict[str, pd.Timestamp]
) -> bool:
    """Download day-ahead prices for a specific country and date chunk"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Special handling for Germany
        if country_name.startswith('Germany'):
            # Skip chunks that span the transition period
            if (
                date_chunk['start'] < GERMANY_HISTORICAL['new']['start_date']
                and date_chunk['end'] >= GERMANY_HISTORICAL['new']['start_date']
            ):
                logging.info(
                    f"Skipping transition period chunk for Germany: "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return True
            
            country_name, area_code = get_german_config(date_chunk['start'])
        
        if is_chunk_downloaded(
            conn, country_name, 'prices',
            date_chunk['start'], date_chunk['end']
        ):
            logging.info(
                f"Price data exists for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return True
        
        prices = client.query_day_ahead_prices(
            area_code,
            start=date_chunk['start'],
            end=date_chunk['end']
        )
        
        if not prices.empty:
            df = pd.DataFrame({
                'timestamp': prices.index,
                'country': country_name,
                'price': prices.values
            })
            
            return save_to_database(
                conn, df, 'day_ahead_prices',
                country_name, date_chunk, 'prices'
            )
        else:
            logging.warning(
                f"No price data for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return False
            
    except Exception as e:
        logging.error(
            f"Error downloading prices for {country_name} - "
            f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
            f"{date_chunk['end'].strftime('%Y-%m-%d')}: {str(e)}"
        )
        return False
    finally:
        if conn:
            conn.close()


def download_load(
    client: EntsoePandasClient,
    country_name: str,
    area_code: str,
    date_chunk: Dict[str, pd.Timestamp]
) -> bool:
    """Download load data for a specific country and date chunk"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Special handling for Germany
        if country_name.startswith('Germany'):
            # Skip chunks that span the transition period
            if (
                date_chunk['start'] < GERMANY_HISTORICAL['new']['start_date']
                and date_chunk['end'] >= GERMANY_HISTORICAL['new']['start_date']
            ):
                logging.info(
                    f"Skipping transition period chunk for Germany: "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return True
            
            # Get correct configuration for the period
            country_name, area_code = get_german_config(date_chunk['start'])
        
        # Check if chunk already downloaded
        if is_chunk_downloaded(
            conn,
            country_name,
            'load',
            date_chunk['start'],
            date_chunk['end']
        ):
            logging.info(
                f"Load data exists for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return True
        
        # Download load data
        load = client.query_load_and_forecast(
            area_code,
            start=date_chunk['start'],
            end=date_chunk['end']
        )
        
        if not load.empty:
            # Resample to hourly data using mean
            if isinstance(load, pd.Series):
                load = pd.DataFrame(load)
            
            # Ensure the index is timezone-aware
            if load.index.tz is None:
                load.index = load.index.tz_localize(TIMEZONE)
            
            # Resample to hourly frequency
            load = load.resample('h').mean()
            
            # Convert DataFrame to desired format
            df = pd.DataFrame({
                'timestamp': load.index,
                'country': country_name,
                # Use get() to handle missing columns
                'actual_load': load.get('Actual Load', None),
                'forecast_load': load.get('Load Forecast', None)
            })
            
            # Drop rows where both actual and forecast are null
            df = df.dropna(subset=['actual_load', 'forecast_load'], how='all')
            
            if not df.empty:
                # Save to database
                return save_to_database(
                    conn, df, 'load_data',
                    country_name, date_chunk, 'load'
                )
            else:
                start_str = date_chunk['start'].strftime('%Y-%m-%d')
                end_str = date_chunk['end'].strftime('%Y-%m-%d')
                logging.warning(
                    f"No valid load data for {country_name} - "
                    f"{start_str} to {end_str}"
                )
                return False
        else:
            start_str = date_chunk['start'].strftime('%Y-%m-%d')
            end_str = date_chunk['end'].strftime('%Y-%m-%d')
            logging.warning(
                f"No load data for {country_name} - "
                f"{start_str} to {end_str}"
            )
            return False
            
    except Exception as e:
        start_str = date_chunk['start'].strftime('%Y-%m-%d')
        end_str = date_chunk['end'].strftime('%Y-%m-%d')
        logging.error(
            f"Error downloading load data for {country_name} - "
            f"{start_str} to {end_str}: {str(e)}"
        )
        return False
    finally:
        if conn:
            conn.close()


def download_generation(
    client: EntsoePandasClient,
    country_name: str,
    area_code: str,
    date_chunk: Dict[str, pd.Timestamp]
) -> bool:
    """Download generation data for a specific country and date chunk"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Special handling for Germany
        if country_name.startswith('Germany'):
            # Skip chunks that span the transition period
            if (
                date_chunk['start'] < GERMANY_HISTORICAL['new']['start_date']
                and date_chunk['end'] >= GERMANY_HISTORICAL['new']['start_date']
            ):
                logging.info(
                    f"Skipping transition period chunk for Germany: "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return True
            
            # Get correct configuration for the period
            country_name, area_code = get_german_config(date_chunk['start'])
        
        # Check if chunk already downloaded
        if is_chunk_downloaded(
            conn,
            country_name,
            'generation',
            date_chunk['start'],
            date_chunk['end']
        ):
            logging.info(
                f"Generation data exists for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return True
        
        # Download generation data
        gen_data = client.query_generation(
            area_code,
            start=date_chunk['start'],
            end=date_chunk['end'],
            psr_type=None  # Get all generation types
        )
        
        if not gen_data.empty:
            # Process the data into the correct format
            processed_df = process_generation_data(gen_data, country_name)
            
            if not processed_df.empty:
                # Save to database
                return save_to_database(
                    conn, processed_df, 'generation_data',
                    country_name, date_chunk, 'generation'
                )
            else:
                logging.warning(
                    f"No valid generation data for {country_name} - "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return False
        else:
            logging.warning(
                f"No generation data for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return False
            
    except Exception as e:
        logging.error(
            f"Error downloading generation data for {country_name} - "
            f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
            f"{date_chunk['end'].strftime('%Y-%m-%d')}: {str(e)}"
        )
        return False
    finally:
        if conn:
            conn.close()


def download_entsoe_data(
    api_key: str,
    start_year: int = 2015,
    end_year: int = 2024,
    max_workers: int = 3,
    countries: List[str] = None,  # Optional list of countries
    data_types: List[str] = None  # Optional list of data types
):
    """
    Download ENTSO-E data using parallel processing
    
    Args:
        api_key: ENTSO-E API key
        start_year: Start year for data download (default: 2015)
        end_year: End year for data download (default: 2024)
        max_workers: Number of parallel workers (default: 3)
        countries: List of countries to download data for. 
                  Options: ['France', 'Spain', 'Italy', 'Germany', 
                           'Switzerland', 'Belgium']
                  If None, downloads for all countries.
        data_types: List of data types to download.
                   Options: ['prices', 'load', 'generation']
                   If None, downloads all data types.
    """
    # Validate and process country input
    available_countries = list(ENTSOE_COUNTRY_CODES.keys()) + ['Germany']
    if countries:
        countries = [c for c in countries if c in available_countries]
        if not countries:
            raise ValueError(
                f"No valid countries specified. Available options: "
                f"{available_countries}"
            )
    else:
        countries = available_countries

    # Validate and process data types input
    if data_types:
        data_types = [d for d in data_types if d in ENTSOE_DATA_TYPES]
        if not data_types:
            raise ValueError(
                f"No valid data types specified. Available options: "
                f"{ENTSOE_DATA_TYPES}"
            )
    else:
        data_types = ENTSOE_DATA_TYPES

    # Initialize the ENTSO-E client
    client = EntsoePandasClient(api_key=api_key)
    
    # Initialize database
    init_database()
    
    # Get date chunks
    date_chunks = create_date_chunks(start_year, end_year)
    
    # Create download tasks for regular countries
    tasks = []
    for country in countries:
        # Handle Germany separately
        if country == 'Germany':
            for date_chunk in date_chunks:
                # Use appropriate German configuration based on date
                country_name, area_code = get_german_config(
                    date_chunk['start']
                )
                tasks.extend([
                    (func, client, country_name, area_code, date_chunk)
                    for func in [
                        download_prices if 'prices' in data_types else None,
                        download_load if 'load' in data_types else None,
                        download_generation if 'generation' in data_types
                        else None
                    ] if func is not None
                ])
            continue
        
        # Regular country handling
        area_code = ENTSOE_COUNTRY_CODES[country]
        for date_chunk in date_chunks:
            tasks.extend([
                (func, client, country, area_code, date_chunk)
                for func in [
                    download_prices if 'prices' in data_types else None,
                    download_load if 'load' in data_types else None,
                    download_generation if 'generation' in data_types else None
                ] if func is not None
            ])
    
    # Download data in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(*task)
            futures.append(future)
            # Small delay between submissions to avoid overwhelming the API
            time.sleep(0.5)
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        logging.error("Please ensure ENTSOE_API_KEY is set in your .env file")
        exit(1)
    
    # Download only load and generation data
    download_entsoe_data(
        api_key,
        data_types=['load', 'generation']
    ) 