#!/usr/bin/env python3

import argparse
import logging
import os
import sqlite3

import pandas as pd

from scripts.config import DATA_PATH


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_tables(conn):
    """Get all table names from the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [table[0] for table in cursor.fetchall()]


def convert_sqlite_to_parquet(sqlite_path, output_dir):
    """
    Convert SQLite database tables to Parquet files.

    Args:
        sqlite_path (str): Path to the SQLite database file
        output_dir (str): Directory where Parquet files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Converting SQLite database: {sqlite_path}")
    logger.info(f"Output directory: {output_dir}")

    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_path)

    # Get all tables
    tables = get_tables(conn)
    logger.info(f"Found {len(tables)} tables in the database")

    # Convert each table to parquet
    for table in tables:
        logger.info(f"Converting table: {table}")

        # Read SQLite table into pandas DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

        # Generate output path
        output_path = os.path.join(output_dir, f"{table}.parquet")

        # Save as Parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {table} to {output_path}")

        # Log some basic statistics
        logger.info(
            f"Table {table}: {len(df)} rows, {len(df.columns)} columns"
        )

    conn.close()
    logger.info("Conversion completed successfully")


def main(sqlite_path, output_dir):
    # Convert paths to absolute paths
    sqlite_path = os.path.abspath(sqlite_path)
    output_dir = os.path.abspath(output_dir)

    # Verify input file exists
    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database file not found: {sqlite_path}")
        return 1

    try:
        convert_sqlite_to_parquet(sqlite_path, output_dir)
        return 0
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        return 1


if __name__ == "__main__":
    sqlite_path = os.path.join(DATA_PATH, "entsoe_data.sqlite")
    output_dir = os.path.join(DATA_PATH, "parquet")
    main(sqlite_path, output_dir)
