"""
Configuration settings for the Power Price Forecasting API.
"""
from pydantic_settings import BaseSettings
from typing import Optional, Dict
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()


# BigQuery
GCP_PROJECT = os.getenv('GCP_PROJECT')
BQ_REGION = os.getenv('BQ_REGION')
BQ_DATASET = os.getenv('BQ_DATASET')
BQ_TABLE_PRICES = os.getenv('BQ_TABLE_PRICES')
BG_TABLE_FLOWS = os.getenv('BG_TABLE_FLOWS')
BG_TABLE_GENERATION = os.getenv('BG_TABLE_GENERATION')
BQ_TABLE_LOAD = os.getenv('BQ_TABLE_LOAD')
BQ_TABLE_WIND_SOLAR = os.getenv('BQ_TABLE_WIND_SOLAR')

DOCKER_IMAGE=os.getenv('DOCKER_IMAGE')

AR_REPO_NAME=os.getenv('AR_REPO_NAME')
AR_REPO_REGION=os.getenv('AR_REPO_REGION')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
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

# Time settings
TIMEZONE = "Europe/Paris"

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY
SECONDS_PER_YEAR_LEAP = 366 * SECONDS_PER_DAY  # Accounting for leap years
SECONDS_PER_YEAR_NON_LEAP = 365 * SECONDS_PER_DAY # Accounting for non-leap years

# Configuration parameters
PREPROCESSING_CONFIG_MLP = {
    # Price outlier threshold in EUR/MWh
    'PRICE_OUTLIER_THRESHOLD': 1000.0,
    'MONTHLY_PRICE_OUTLIER_THRESHOLD': 300.0,
    
    # Feature engineering
    'LAG_HOURS': 24*3,  # Default lag hours (3 days)
    
    # Data splitting - chronological split
    'VAL_SIZE': 0.2,  # 20% of data (by date) for validation
    'TEST_SIZE': 0.2,  # Last 20% of data (by date) for testing
}

PREPROCESSING_CONFIG_ATT = {
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

class Settings(BaseSettings):
    """API Configuration Settings"""
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Power Price Forecasting API"
    
    # Deployment Settings
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"  # Allow extra fields in environment variables

# Will raise an error if ENTSOE_API_KEY is not set
settings = Settings()