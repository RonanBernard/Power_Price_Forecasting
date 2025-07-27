import sqlite3
import pandas as pd
from pathlib import Path
from scripts.download_entsoe_data import COUNTRY_CODES


# Get project root directory (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent
DB_FILE = PROJECT_ROOT / 'data' / 'entsoe_data.sqlite'


def check_database():
    """Check database contents and download log"""
    conn = sqlite3.connect(DB_FILE)
    
    # Check prices table
    prices_df = pd.read_sql("""
        SELECT 
            country,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            COUNT(*) as total_records
        FROM day_ahead_prices
        GROUP BY country
    """, conn)
    
    print("\nPrice Data Summary:")
    print(prices_df)
    
    # Check download log
    log_df = pd.read_sql("""
        SELECT 
            country,
            data_type,
            MIN(start_date) as first_chunk,
            MAX(end_date) as last_chunk,
            COUNT(*) as chunks_downloaded
        FROM download_log
        WHERE status = 'success'
        GROUP BY country, data_type
    """, conn)
    
    print("\nDownload Log Summary:")
    print(log_df)
    
    conn.close()


def clear_download_log(countries=None, data_types=None):
    """
    Clear download log entries to force re-download.
    
    Args:
        countries: List of countries to clear log for. If None, clears all.
        data_types: List of data types to clear log for. If None, clears all.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    base_query = "DELETE FROM download_log WHERE status = 'success'"
    conditions = []
    params = []
    
    if countries:
        conditions.append("country IN (" + ",".join("?" * len(countries)) + ")")
        params.extend(countries)
    
    if data_types:
        conditions.append("data_type IN (" + ",".join("?" * len(data_types)) + ")")
        params.extend(data_types)
    
    if conditions:
        query = base_query + " AND " + " AND ".join(conditions)
    else:
        query = base_query
    
    cursor.execute(query, params)
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"Cleared {deleted_count} log entries.")


def delete_price_data(countries=None):
    """
    Delete existing price data for specified countries.
    
    Args:
        countries: List of countries to delete data for. If None, deletes all.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if countries:
        placeholders = ",".join("?" * len(countries))
        query = f"DELETE FROM day_ahead_prices WHERE country IN ({placeholders})"
        cursor.execute(query, countries)
    else:
        cursor.execute("DELETE FROM day_ahead_prices")
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"Deleted {deleted_count} price records.")


def find_price_gaps(countries=None):
    """
    Find gaps in price data for specified countries.
    Returns a dictionary of countries with their missing date ranges.
    
    Args:
        countries: List of countries to check. If None, checks all.
    """
    conn = sqlite3.connect(DB_FILE)
    
    # First get all available data points
    query = """
        SELECT country, timestamp
        FROM day_ahead_prices
        WHERE country = ?
        ORDER BY timestamp
    """
    
    gaps = {}
    
    for country in (countries or COUNTRY_CODES.keys()):
        df = pd.read_sql_query(query, conn, params=[country])
        if df.empty:
            print(f"No data found for {country}")
            continue
            
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Expected hourly timestamps from 2015 to 2024
        expected_range = pd.date_range(
            start='2015-01-01',
            end='2024-12-31 23:00:00',
            freq='H',
            tz='Europe/Paris'
        )
        
        # Find missing timestamps
        actual_times = set(df['timestamp'])
        missing_times = [t for t in expected_range if t not in actual_times]
        
        if missing_times:
            # Group consecutive missing times into ranges
            gaps[country] = []
            start = missing_times[0]
            prev = start
            
            for curr in missing_times[1:]:
                if curr - prev > pd.Timedelta(hours=1):
                    gaps[country].append((start, prev))
                    start = curr
                prev = curr
            
            gaps[country].append((start, prev))
            
            # Print summary
            print(f"\nGaps found for {country}:")
            for start, end in gaps[country]:
                print(f"  Missing: {start} to {end}")
        else:
            print(f"\nNo gaps found for {country}")
    
    conn.close()
    return gaps


if __name__ == "__main__":
    # First check current state
    print("Current database state:")
    check_database()
    
    # Find gaps in price data
    print("\nChecking for gaps in price data...")
    gaps = find_price_gaps(['France', 'Spain'])
    
    if not any(gaps.values()):
        print("\nNo gaps found in the data. No need to re-download.") 