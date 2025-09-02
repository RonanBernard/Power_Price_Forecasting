"""
Configuration file containing all constant variables used across the project.
"""
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

ENTSOE_API_KEY = os.getenv('ENTSOE_API_KEY')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODELS_PATH = PROJECT_ROOT / 'models'
LOGS_PATH = MODELS_PATH / 'logs'
API_PATH = PROJECT_ROOT / 'api'
API_MODELS_PATH = API_PATH / 'models'

# Country mappings for preprocessing
COUNTRY_DICT = {
    'France': 'FR',
    'Germany': 'DE',
    'Germany_Luxembourg': 'DE',
    'Germany_Austria_Luxembourg': 'DE',
    'Italy': 'IT',
    'Italy_North': 'IT',
    'Spain': 'ES',
    'Great Britain': 'GB',
    'Netherlands': 'NL',
    'Belgium': 'BE',
    'Switzerland': 'CH',
    'Austria': 'AT',
}

# ENTSOE API country codes
ENTSOE_COUNTRY_CODES = {
    'France': '10YFR-RTE------C',
    'Spain': '10YES-REE------0',
    'Italy': '10Y1001A1001A73I',  # IT_NORD bidding zone
    'Switzerland': '10YCH-SWISSGRIDZ',
    'Belgium': '10YBE----------2'
}

# Country code mappings for crossborder flows (using short codes)
CROSSBORDER_COUNTRY_CODES = {
    'France': 'FR',
    'Spain': 'ES',
    'Italy': 'IT',
    'Switzerland': 'CH',
    'Belgium': 'BE',
    'Germany': 'DE',
    'Great Britain': 'GB'
}

# Special handling for Germany due to bidding zone change
GERMANY_HISTORICAL = {
    'old': {
        'name': 'Germany',  # Simplified name
        'code': '10Y1001A1001A63L',  # DE_AT_LU
        'end_date': pd.Timestamp('2018-09-30', tz='Europe/Berlin')
    },
    'new': {
        'name': 'Germany',  # Simplified name
        'code': '10Y1001A1001A82H',  # DE_LU
        'start_date': pd.Timestamp('2018-10-01', tz='Europe/Berlin')
    }
}

# ENTSOE data types
ENTSOE_DATA_TYPES = [
    'prices', 'load', 'generation', 'wind_solar_forecast', 'crossborder_flows'
]

# ENTSOE generation types
ENTSOE_GENERATION_TYPES = [
    'Biomass',
    'Fossil Gas',
    'Fossil Hard coal',
    'Fossil Oil',
    'Hydro Run-of-river and poundage',
    'Hydro Water Reservoir',
    'Nuclear',
    'Solar',
    'Waste',
    'Wind Onshore',
    'Fossil Brown coal/Lignite',
    'Wind Offshore',
    'Other',
    'Geothermal',
    'Marine',
    'Other renewable',
    'Hydro Pumped Storage_Generation',
    'Hydro Pumped Storage_Consumption'
]

# ENTSOE download settings
ENTSOE_CHUNK_SIZE = 90  # 3 months in days

# Price data files
FUEL_PRICES_FILES = {
    'eua': (
        'European Union Allowance (EUA) '
        'Yearly Futures Historical Data.csv'
    ),
    'ttf': (
        'ICE Dutch TTF Natural Gas '
        'Futures Historical Data.csv'
    ),
    'ara': (
        'Coal (API2) CIF ARA (ARGUS-McCloskey) '
        'Futures Historical Data.csv'
    ),
    'fx': 'USD_EUR Historical Data.csv'
}

# Database settings
DB_FILE = DATA_PATH / 'entsoe_data.sqlite'

# Time settings
TIMEZONE = "Europe/Paris"

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
SECONDS_PER_YEAR_LEAP = 366 * SECONDS_PER_DAY  # Accounting for leap years
SECONDS_PER_YEAR_NON_LEAP = 365 * SECONDS_PER_DAY # Accounting for non-leap years

# Configuration parameters
PREPROCESSING_CONFIG_V1 = {
    # Price outlier threshold in EUR/MWh
    'PRICE_OUTLIER_THRESHOLD': 1000.0,
    'MONTHLY_PRICE_OUTLIER_THRESHOLD': 300.0,
    
    # Feature engineering
    'LAG_HOURS': 24*3,  # Default lag hours (3 days)
    
    # Data splitting - chronological split
    'VAL_SIZE': 0.2,  # 20% of data (by date) for validation
    'TEST_SIZE': 0.2,  # Last 20% of data (by date) for testing
}

PREPROCESSING_CONFIG_V2 = {
    # Price outlier threshold in EUR/MWh
    'PRICE_OUTLIER_THRESHOLD': 1000.0,
    'MONTHLY_PRICE_OUTLIER_THRESHOLD': 300.0,
    
    # Feature engineering
    'HISTORY_HOURS': 168,  # 1 week of hourly data
    'HORIZON_HOURS': 24,  # Predict next 24 hours
    'STRIDE_HOURS': 24,  # Stride between samples
    
    # Data splitting - chronological split
    'VAL_SIZE': 0.2,  # 20% of data (by date) for validation
    'TEST_SIZE': 0.2,  # Last 20% of data (by date) for testing
    'CV': 5,  # Number of folds for cross-validation
    'ROLLING_HORIZON_VAL_SIZE': 0.2,  # Last 20% of data (by date) for testing
}

PREPROCESSING_CONFIG_V3 = {
    # Price outlier threshold in EUR/MWh
    'PRICE_OUTLIER_THRESHOLD': 1000.0,
    'MONTHLY_PRICE_OUTLIER_THRESHOLD': 300.0,
    
    # Feature engineering
    'HISTORY_HOURS': 168,  # 1 week of hourly data
    'HORIZON_HOURS': 24,  # Predict next 24 hours
    'STRIDE_HOURS': 24,  # Stride between samples
    
    # Data splitting - chronological split
    'VAL_SIZE': 0.2,  # 20% of data (by date) for validation
    'TEST_SIZE': 0.2,  # Last 20% of data (by date) for testing
    'CV': 5,  # Number of folds for cross-validation
    'ROLLING_HORIZON_VAL_SIZE': 0.2,  # Last 20% of data (by date) for testing
}

DEFAULT_FUEL_PRICES = {
    'TTF_EUR': 30.8,
    'EUA_EUR': 70,
    'ARA_USD': 99.5,
    'USD_EUR': 0.86
}