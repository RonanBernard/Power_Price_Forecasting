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


def find_gaps(country=None, data_type=None):
    """
    Find gaps in data for specified country and data type.
    If no country or data_type specified, checks all combinations.
    Returns a list of (country/pair, data_type, missing_count, total_expected) 
    tuples.
    
    Args:
        country: Country to check, if None checks all countries
        data_type: One of 'prices', 'load', 'generation', or 'crossborder_flows',
                   if None checks all
    """
    # Map legacy country names to standard names for crossborder flows
    country_mapping = {
        'Italy_North': 'Italy',
        'Germany_Austria_Luxembourg': 'Germany',
        'Germany_Luxembourg': 'Germany'
    }
    conn = sqlite3.connect(DB_FILE)
    
    # Map data types to table info
    table_map = {
        'prices': ('day_ahead_prices', 'timestamp'),
        'load': ('load_data', 'timestamp'),
        'generation': ('generation_data', 'timestamp'),
        'crossborder_flows': ('crossborder_flows', 'timestamp')
    }
    
    if data_type and data_type not in table_map:
        raise ValueError(
            f"Invalid data type. Must be one of: {list(table_map.keys())}"
        )
    
    # Get all countries if none specified
    if country is None:
        cursor = conn.cursor()
        countries = set()
        for data_type_name, (table, _) in table_map.items():
            if data_type_name == 'crossborder_flows':
                # For crossborder flows, get countries from both
                # country_from and country_to
                cursor.execute(f"SELECT DISTINCT country_from FROM {table}")
                countries.update(row[0] for row in cursor.fetchall())
                cursor.execute(f"SELECT DISTINCT country_to FROM {table}")
                countries.update(row[0] for row in cursor.fetchall())
            else:
                # For other tables, use the standard country column
                cursor.execute(f"SELECT DISTINCT country FROM {table}")
                countries.update(row[0] for row in cursor.fetchall())
        countries = sorted(list(countries))
    else:
        # Map the country name if it exists in the mapping
        mapped_country = country_mapping.get(country, country)
        countries = [mapped_country]
    
    # Get all data types if none specified
    data_types = [data_type] if data_type else list(table_map.keys())
    
    all_gaps = []
    
    for curr_country in countries:
        for curr_type in data_types:
            table, timestamp_col = table_map[curr_type]
            
            # Special handling for crossborder flows
            if curr_type == 'crossborder_flows':
                # Map the current country if needed
                mapped_country = country_mapping.get(curr_country, curr_country)
                
                # Get all country pairs for the current country
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT DISTINCT country_from, country_to
                    FROM {table}
                    WHERE country_from = ? OR country_to = ?
                    ORDER BY country_from, country_to
                """, [mapped_country, mapped_country])
                country_pairs = cursor.fetchall()
                
                if not country_pairs:
                    print(f"No {curr_type} data found for {curr_country}")
                    continue
                
                # Check gaps for each country pair
                for country_from, country_to in country_pairs:
                    pair_name = f"{country_from}->{country_to}"
                    
                    # Get available data points for this pair
                    query = f"""
                        SELECT {timestamp_col}
                        FROM {table}
                        WHERE country_from = ? AND country_to = ?
                        ORDER BY {timestamp_col}
                    """
                    
                    df = pd.read_sql_query(query, conn, params=[country_from, country_to])
                    
                    if df.empty:
                        print(f"No {curr_type} data found for {pair_name}")
                        continue
                    
                    # Convert timestamps to datetime
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
                    
                    # Expected hourly timestamps from 2015 to 2024
                    expected_range = pd.date_range(
                        start='2015-01-01',
                        end='2024-12-31 23:00:00',
                        freq='h',
                        tz='Europe/Paris'
                    )
                    
                    # Find missing timestamps
                    actual_times = set(df[timestamp_col])
                    missing_times = [t for t in expected_range if t not in actual_times]
                    total_expected = len(expected_range)
                    missing_count = len(missing_times)
                    
                    if missing_count == 0:
                        print(f"No gaps found for {pair_name} - {curr_type}")
                        continue
                    
                    all_gaps.append((pair_name, curr_type, missing_count, total_expected))
                    
                    # Print summary
                    print(f"{pair_name} - {curr_type}: {missing_count}/{total_expected} missing timestamps")
            else:
                # Standard handling for other data types
                # Get all available data points
                query = f"""
                    SELECT {timestamp_col}
                    FROM {table}
                    WHERE country = ?
                    ORDER BY {timestamp_col}
                """
                
                df = pd.read_sql_query(query, conn, params=[curr_country])
                
                if df.empty:
                    print(f"No {curr_type} data found for {curr_country}")
                    continue
                
                # Convert timestamps to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
                
                # Expected hourly timestamps from 2015 to 2024
                expected_range = pd.date_range(
                    start='2015-01-01',
                    end='2024-12-31 23:00:00',
                    freq='h',
                    tz='Europe/Paris'
                )
                
                # Find missing timestamps
                actual_times = set(df[timestamp_col])
                missing_times = [t for t in expected_range if t not in actual_times]
                total_expected = len(expected_range)
                missing_count = len(missing_times)
                
                if missing_count == 0:
                    print(f"No gaps found for {curr_country} - {curr_type}")
                    continue
                
                all_gaps.append((curr_country, curr_type, missing_count, total_expected))
                
                # Print summary
                print(f"{curr_country} - {curr_type}: {missing_count}/{total_expected} missing timestamps")
    
    conn.close()
    return all_gaps


def delete_data(country=None, data_type=None):
    """
    Delete data for specified country and data type.
    If no country or data_type specified, deletes all data.
    Also removes corresponding download logs.
    
    Args:
        country: Country to delete data for, if None deletes all countries
        data_type: One of 'prices', 'load', 'generation', or 'crossborder_flows',
                   if None deletes all
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Map data types to table names
    table_map = {
        'prices': 'day_ahead_prices',
        'load': 'load_data',
        'generation': 'generation_data',
        'wind_solar_forecast': 'wind_solar_forecast',
        'crossborder_flows': 'crossborder_flows'
    }
    
    if data_type and data_type not in table_map:
        raise ValueError(
            f"Invalid data type. Must be one of: {list(table_map.keys())}"
        )
    
    # Get tables to delete from
    tables = [table_map[data_type]] if data_type else list(table_map.values())
    
    total_deleted = 0
    
    # Delete data from tables
    for table in tables:
        if country:
            if table == 'crossborder_flows':
                # For crossborder flows, delete records where country is either
                # country_from or country_to
                query = f"DELETE FROM {table} WHERE country_from = ? OR country_to = ?"
                cursor.execute(query, [country, country])
            else:
                # For other tables, use the standard country column
                query = f"DELETE FROM {table} WHERE country = ?"
                cursor.execute(query, [country])
        else:
            query = f"DELETE FROM {table}"
            cursor.execute(query)
        
        deleted_count = cursor.rowcount
        total_deleted += deleted_count
        
        if country:
            print(f"Deleted {deleted_count} records from {table} for {country}")
        else:
            print(f"Deleted {deleted_count} records from {table}")
    
    # Delete corresponding download logs
    if country and data_type:
        cursor.execute(
            "DELETE FROM download_log WHERE country = ? AND data_type = ?",
            [country, data_type]
        )
    elif country:
        cursor.execute("DELETE FROM download_log WHERE country = ?", [country])
    elif data_type:
        cursor.execute(
            "DELETE FROM download_log WHERE data_type = ?", [data_type]
        )
    else:
        cursor.execute("DELETE FROM download_log")
    
    log_deleted = cursor.rowcount
    print(f"Deleted {log_deleted} records from download_log")
    
    conn.commit()
    conn.close()
    
    return total_deleted


if __name__ == "__main__":
    # First check current state
    print("Current database state:")
    check_database()
    
    # Interactive mode for checking gaps
    print("\nWould you like to check for gaps in the data?")
    response = input("Enter 'y' to continue: ")
    
    if response.lower() == 'y':
        country = input("Enter country code (or press Enter for all): ").strip()
        print("\nAvailable data types: prices, load, generation, " +
              "crossborder_flows")
        data_type = input("Enter data type (or press Enter for all): ").strip().lower()
        
        # Convert empty strings to None
        country = country if country else None
        data_type = data_type if data_type else None
        
        gaps = find_gaps(country, data_type)
        
        if gaps:
            print("\nWould you like to delete this data and re-download?")
            delete_resp = input("Enter 'y' to delete: ")
            if delete_resp.lower() == 'y':
                delete_data(country, data_type)
                clear_download_log(
                    countries=[country] if country else None,
                    data_types=[data_type] if data_type else None
                ) 