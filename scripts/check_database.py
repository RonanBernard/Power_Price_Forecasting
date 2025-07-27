import sqlite3
import pandas as pd
from pathlib import Path


# Get project root directory (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent
DB_FILE = PROJECT_ROOT / 'data' / 'entsoe_data.sqlite'


def check_database():
    """Check database contents and download log"""
    conn = sqlite3.connect(DB_FILE)
    
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


def check_data(data_type):
    """
    Get summary for a specific data type.
    
    Args:
        data_type: One of 'prices', 'load', or 'generation'
    """
    conn = sqlite3.connect(DB_FILE)
    
    if data_type == 'prices':
        df = pd.read_sql("""
            SELECT 
                country,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(*) as total_records
            FROM day_ahead_prices
            GROUP BY country
        """, conn)
        print("\nPrice Data Summary:")
        
    elif data_type == 'load':
        df = pd.read_sql("""
            SELECT 
                country,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(*) as total_records,
                SUM(CASE WHEN actual_load IS NOT NULL 
                    THEN 1 ELSE 0 END) as actual_records,
                SUM(CASE WHEN forecast_load IS NOT NULL 
                    THEN 1 ELSE 0 END) as forecast_records
            FROM load_data
            GROUP BY country
        """, conn)
        print("\nLoad Data Summary:")
        
    elif data_type == 'generation':
        df = pd.read_sql("""
            SELECT 
                country,
                generation_type,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date,
                COUNT(*) as total_records
            FROM generation_data
            GROUP BY country, generation_type
        """, conn)
        print("\nGeneration Data Summary:")
    
    print(df)
    conn.close()
    return df


def find_gaps(country, data_type):
    """
    Find gaps in data for a specific country and data type.
    Returns a list of (start, end) tuples representing gaps.
    
    Args:
        country: Country to check
        data_type: One of 'prices', 'load', or 'generation'
    """
    conn = sqlite3.connect(DB_FILE)
    
    # Map data types to table info
    table_map = {
        'prices': ('day_ahead_prices', 'timestamp'),
        'load': ('load_data', 'timestamp'),
        'generation': ('generation_data', 'timestamp')
    }
    
    if data_type not in table_map:
        raise ValueError(
            f"Invalid data type. Must be one of: {list(table_map.keys())}"
        )
    
    table, timestamp_col = table_map[data_type]
    
    # Get all available data points
    query = f"""
        SELECT {timestamp_col}
        FROM {table}
        WHERE country = ?
        ORDER BY {timestamp_col}
    """
    
    df = pd.read_sql_query(query, conn, params=[country])
    conn.close()
    
    if df.empty:
        print(f"No {data_type} data found for {country}")
        return []
    
    # Convert timestamps to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Expected hourly timestamps from 2015 to 2024
    expected_range = pd.date_range(
        start='2015-01-01',
        end='2024-12-31 23:00:00',
        freq='H',
        tz='Europe/Paris'
    )
    
    # Find missing timestamps
    actual_times = set(df[timestamp_col])
    missing_times = [t for t in expected_range if t not in actual_times]
    
    if not missing_times:
        print(f"\nNo gaps found for {country} - {data_type}")
        return []
    
    # Group consecutive missing times into ranges
    gaps = []
    start = missing_times[0]
    prev = start
    
    for curr in missing_times[1:]:
        if curr - prev > pd.Timedelta(hours=1):
            gaps.append((start, prev))
            start = curr
        prev = curr
    
    gaps.append((start, prev))
    
    # Print summary
    print(f"\nGaps found for {country} - {data_type}:")
    for start, end in gaps:
        print(f"  Missing: {start} to {end}")
    
    return gaps


def delete_data(country, data_type):
    """
    Delete data for a specific country and data type.
    
    Args:
        country: Country to delete data for
        data_type: One of 'prices', 'load', or 'generation'
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Map data types to table names
    table_map = {
        'prices': 'day_ahead_prices',
        'load': 'load_data',
        'generation': 'generation_data'
    }
    
    if data_type not in table_map:
        raise ValueError(
            f"Invalid data type. Must be one of: {list(table_map.keys())}"
        )
    
    table = table_map[data_type]
    query = f"DELETE FROM {table} WHERE country = ?"
    
    cursor.execute(query, [country])
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"Deleted {deleted_count} records from {table} for {country}")
    return deleted_count


if __name__ == "__main__":
    # First check current state
    print("Current database state:")
    check_database()
    
    # Interactive mode for checking gaps
    print("\nWould you like to check for gaps in the data?")
    response = input("Enter 'y' to continue: ")
    
    if response.lower() == 'y':
        country = input("Enter country code (e.g., France): ").strip()
        print("\nAvailable data types: prices, load, generation")
        data_type = input("Enter data type: ").strip().lower()
        
        if country and data_type:
            gaps = find_gaps(country, data_type)
            
            if gaps:
                print("\nWould you like to delete this data and re-download?")
                delete_resp = input("Enter 'y' to delete: ")
                if delete_resp.lower() == 'y':
                    delete_data(country, data_type)
                    clear_download_log(
                        countries=[country],
                        data_types=[data_type]
                    ) 