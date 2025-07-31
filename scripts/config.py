"""
Configuration file containing all constant variables used across the project.
"""
from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'

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
ENTSOE_DATA_TYPES = ['prices', 'load', 'generation', 'wind_solar_forecast']

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