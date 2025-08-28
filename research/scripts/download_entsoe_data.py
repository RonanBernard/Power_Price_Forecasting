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
from tqdm import tqdm
from research.scripts.check_database import delete_data
try:
    from .config import (
        DATA_PATH,
        TIMEZONE,
        ENTSOE_COUNTRY_CODES,
        CROSSBORDER_COUNTRY_CODES,
        GERMANY_HISTORICAL,
        ENTSOE_DATA_TYPES,
        ENTSOE_GENERATION_TYPES,
        ENTSOE_CHUNK_SIZE,
        DB_FILE
    )
except ImportError:
    from config import (
        DATA_PATH,
        TIMEZONE,
        ENTSOE_COUNTRY_CODES,
        CROSSBORDER_COUNTRY_CODES,
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
            
            # Create table for wind and solar forecasts
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS wind_solar_forecast (
                timestamp DATETIME,
                country TEXT,
                forecast_type TEXT,
                value FLOAT,
                unit TEXT DEFAULT 'MW',
                PRIMARY KEY (timestamp, country, forecast_type)
            )
            ''')
            
            # Create table for crossborder flows
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS crossborder_flows (
                timestamp DATETIME,
                country_from TEXT,
                country_to TEXT,
                flow FLOAT,
                unit TEXT DEFAULT 'MW',
                PRIMARY KEY (timestamp, country_from, country_to)
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
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_forecast_time_country_type
            ON wind_solar_forecast(timestamp, country, forecast_type)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_crossborder_time_countries
            ON crossborder_flows(timestamp, country_from, country_to)
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
            
            # Debug: print basic information about the load data
            if isinstance(load, pd.DataFrame):
                logging.info(f"Country: {country_name}")
                logging.info(f"Load data shape: {load.shape}")
                logging.info(f"Load data columns: {load.columns.tolist()}")
            else:
                logging.info(f"Load data length: {len(load)}")
            
            # Convert DataFrame to desired format
            df = pd.DataFrame({
                'timestamp': load.index,
                'country': country_name,
                # Use get() to handle missing columns
                # Use exact column names from the API
                'actual_load': load['Actual Load'] if isinstance(load, pd.DataFrame) else load,
                'forecast_load': load['Forecasted Load'] if isinstance(load, pd.DataFrame) else None
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


def download_wind_solar_forecast(
    client: EntsoePandasClient,
    country_name: str,
    area_code: str,
    date_chunk: Dict[str, pd.Timestamp]
) -> bool:
    """Download wind and solar forecast data for a specific country and date chunk"""
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
        
        # Check if chunk already downloaded
        if is_chunk_downloaded(
            conn,
            country_name,
            'wind_solar_forecast',
            date_chunk['start'],
            date_chunk['end']
        ):
            logging.info(
                f"Wind/Solar forecast exists for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return True
        
        # Download forecast data
        forecast = client.query_wind_and_solar_forecast(
            area_code,
            start=date_chunk['start'],
            end=date_chunk['end']
        )
        
        if not forecast.empty:
            # Debug: print basic information about the forecast data
            if isinstance(forecast, pd.DataFrame):
                available_types = forecast.columns.tolist()
                missing_types = []
                expected_types = ['Solar', 'Wind Onshore', 'Wind Offshore']
                
                for type_ in expected_types:
                    if type_ not in available_types:
                        missing_types.append(type_)
                
                logging.info(
                    f"{country_name} forecast data - "
                    f"Available: {available_types}, "
                    f"Missing: {missing_types}"
                )
            else:
                logging.info(f"Unexpected format for {country_name} forecast")
            
            # Ensure the index is timezone-aware
            if forecast.index.tz is None:
                forecast.index = forecast.index.tz_localize(TIMEZONE)
            
            # Resample to hourly frequency
            forecast = forecast.resample('h').mean()
            
            # Convert to the format for database storage
            records = []
            for ts in forecast.index:
                for col in forecast.columns:
                    value = forecast.loc[ts, col]
                    if pd.notna(value):
                        records.append({
                            'timestamp': ts,
                            'country': country_name,
                            'forecast_type': col,
                            'value': value
                        })
            
            if records:
                df = pd.DataFrame(records)
                # Save to database
                return save_to_database(
                    conn,
                    df,
                    'wind_solar_forecast',
                    country_name,
                    date_chunk,
                    'wind_solar_forecast'
                )
            else:
                logging.warning(
                    f"No valid forecast data for {country_name} - "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return False
        else:
            logging.warning(
                f"No forecast data for {country_name} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return False
            
    except Exception as e:
        logging.error(
            f"Error downloading wind/solar forecast for {country_name} - "
            f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
            f"{date_chunk['end'].strftime('%Y-%m-%d')}: {str(e)}"
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


def download_crossborder_flows(
    client: EntsoePandasClient,
    country_from: str,
    country_to: str,
    date_chunk: Dict[str, pd.Timestamp]
) -> bool:
    """Download crossborder flows data between two countries for a 
    specific date chunk"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Check if chunk already downloaded
        if is_chunk_downloaded(
            conn,
            f"{country_from}-{country_to}",
            'crossborder_flows',
            date_chunk['start'],
            date_chunk['end']
        ):
            logging.info(
                f"Crossborder flows data exists for "
                f"{country_from}->{country_to} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return True
        
        # Get country codes for the API call
        country_from_code = CROSSBORDER_COUNTRY_CODES.get(country_from)
        country_to_code = CROSSBORDER_COUNTRY_CODES.get(country_to)
        
        if not country_from_code or not country_to_code:
            logging.error(
                f"Country codes not found for {country_from} or {country_to}"
            )
            return False
        
        # Download crossborder flows data
        flows = client.query_crossborder_flows(
            country_from_code,
            country_to_code,
            start=date_chunk['start'],
            end=date_chunk['end']
        )
        
        if not flows.empty:
            # Ensure the index is timezone-aware
            if flows.index.tz is None:
                flows.index = flows.index.tz_localize(TIMEZONE)
            
            # Resample to hourly frequency
            flows = flows.resample('h').mean()
            
            # Convert to DataFrame format for database storage
            df = pd.DataFrame({
                'timestamp': flows.index,
                'country_from': country_from,
                'country_to': country_to,
                'flow': flows.values
            })
            
            # Drop rows with null flows
            df = df.dropna(subset=['flow'])
            
            if not df.empty:
                # Save to database
                return save_to_database(
                    conn, df, 'crossborder_flows',
                    f"{country_from}-{country_to}", 
                    date_chunk, 'crossborder_flows'
                )
            else:
                logging.warning(
                    f"No valid crossborder flows data for "
                    f"{country_from}->{country_to} - "
                    f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                    f"{date_chunk['end'].strftime('%Y-%m-%d')}"
                )
                return False
        else:
            logging.warning(
                f"No crossborder flows data for "
                f"{country_from}->{country_to} - "
                f"{date_chunk['start'].strftime('%Y-%m-%d')} to "
                f"{date_chunk['end'].strftime('%Y-%m-%d')}"
            )
            return False
            
    except Exception as e:
        logging.error(
            f"Error downloading crossborder flows for "
            f"{country_from}->{country_to} - "
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
                        download_generation if 'generation' in data_types else None,
                        download_wind_solar_forecast if 'wind_solar_forecast' in data_types else None
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
                    download_generation if 'generation' in data_types else None,
                    download_wind_solar_forecast if 'wind_solar_forecast' in data_types else None
                ] if func is not None
            ])
    
    # Add crossborder flows tasks if requested
    if 'crossborder_flows' in data_types:
        # Define France's neighboring countries for crossborder flows
        france_neighbors = [
            'Spain', 'Italy', 'Switzerland', 'Belgium', 'Germany', 'Great Britain'
        ]
        
        for neighbor in france_neighbors:
            if neighbor in CROSSBORDER_COUNTRY_CODES:
                for date_chunk in date_chunks:
                    # Add both directions: France -> Neighbor and Neighbor -> France
                    tasks.extend([
                        (download_crossborder_flows, client, 'France', 
                         neighbor, date_chunk),
                        (download_crossborder_flows, client, neighbor, 
                         'France', date_chunk)
                    ])
    
    # Calculate total tasks for progress bar
    total_tasks = len(tasks)
    completed_tasks = 0
    
    # Create progress bar
    progress_bar = tqdm(
        total=total_tasks,
        desc="Downloading ENTSOE data",
        unit="chunk"
    )

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
            try:
                result = future.result()
                if result:  # If download was successful
                    completed_tasks += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    completed=f"{completed_tasks}/{total_tasks}"
                )
            except Exception as e:
                logging.error(f"Task failed: {str(e)}")
                progress_bar.update(1)
    
    progress_bar.close()


def query_crossborder_flows_from_db(
    country_from: str = None,
    country_to: str = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Query crossborder flows data from the database
    
    Args:
        country_from: Source country (optional)
        country_to: Destination country (optional)
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
    
    Returns:
        DataFrame with crossborder flows data
    """
    conn = get_db_connection()
    try:
        # Build the SQL query
        query = "SELECT * FROM crossborder_flows WHERE 1=1"
        params = []
        
        if country_from:
            query += " AND country_from = ?"
            params.append(country_from)
        
        if country_to:
            query += " AND country_to = ?"
            params.append(country_to)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp, country_from, country_to"
        
        # Execute query and return DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert timestamp to datetime
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
        
    finally:
        conn.close()


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        logging.error("Please ensure ENTSOE_API_KEY is set in your .env file")
        exit(1)
    
    # Delete existing crossborder flows data
    delete_data(country=None, data_type='crossborder_flows')
    print("Cleared existing crossborder flows data")
    
    # Download only crossborder flows data
    download_entsoe_data(
        api_key,
        start_year=2015,
        end_year=2024,
        data_types=['crossborder_flows']
    )