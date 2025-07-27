# Power Price Forecasting

A Python project for forecasting day-ahead electricity prices on the EPEX (European Power Exchange) market. The project includes data collection from ENTSO-E and price forecasting capabilities.

## Project Objective

The main goal is to develop accurate forecasting models for day-ahead electricity prices on the EPEX market. This involves:
- Collecting historical price data from ENTSO-E
- Gathering related features (load, generation mix)
- Building and evaluating forecasting models
- Providing insights into price dynamics

## Data Collection

The project uses the ENTSO-E Transparency Platform API to collect:
- Day-ahead prices
- Actual load and load forecasts
- Generation mix by source

### Coverage

**Markets/Bidding Zones:**
- France (EPEX Paris)
- Germany (EPEX Frankfurt)
  - Pre Oct 2018: DE-AT-LU zone
  - Post Oct 2018: DE-LU zone
- Belgium (EPEX Brussels)
- Switzerland (EPEX Zurich)
- Spain (OMIE)
- Italy (North zone/IT_NORD)

**Time Period:** 2015-2024

**Data Resolution:** Hourly

## Project Structure

```
Power_Price_Forecasting/
├── data/                    # Data storage directory
│   └── entsoe_data.sqlite  # SQLite database with all collected data
├── scripts/                 # Data collection and processing scripts
│   ├── __init__.py
│   ├── download_entsoe_data.py  # Main data download script
│   └── check_database.py    # Utility to check data completeness
├── requirements.txt         # Python dependencies
└── README.md
```

## Database Structure

The SQLite database (`entsoe_data.sqlite`) contains the following tables:

1. **day_ahead_prices**
   - timestamp (DATETIME)
   - country (TEXT)
   - price (FLOAT)
   - unit (TEXT, default: 'EUR/MWh')
   - Primary Key: (timestamp, country)

2. **load_data**
   - timestamp (DATETIME)
   - country (TEXT)
   - actual_load (FLOAT)
   - forecast_load (FLOAT)
   - unit (TEXT, default: 'MW')
   - Primary Key: (timestamp, country)

3. **generation_data**
   - timestamp (DATETIME)
   - country (TEXT)
   - generation_type (TEXT)
   - value (FLOAT)
   - unit (TEXT, default: 'MW')
   - Primary Key: (timestamp, country, generation_type)

4. **download_log**
   - country (TEXT)
   - data_type (TEXT)
   - start_date (DATETIME)
   - end_date (DATETIME)
   - download_timestamp (DATETIME)
   - status (TEXT)
   - Primary Key: (country, data_type, start_date, end_date)

## Setup and Usage

1. **Environment Setup**
   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv power_env
   source power_env/bin/activate  # Linux/Mac
   # or
   power_env\Scripts\activate     # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **API Configuration**
   - Create a `.env` file in the project root
   - Add your ENTSO-E API key:
     ```
     ENTSOE_API_KEY=your_api_key_here
     ```

3. **Data Download**
   ```python
   from scripts.download_entsoe_data import download_entsoe_data
   from dotenv import load_dotenv
   import os

   load_dotenv()

   # Download all data types for all countries
   download_entsoe_data(api_key=os.getenv("ENTSOE_API_KEY"))

   # Or specify countries and data types
   download_entsoe_data(
       api_key=os.getenv("ENTSOE_API_KEY"),
       countries=['France', 'Germany'],  # Optional: specific countries
       data_types=['prices']            # Optional: specific data types
   )
   ```

4. **Check Data Completeness**
   ```python
   from scripts.check_database import check_database, find_price_gaps

   # View database summary
   check_database()

   # Check for gaps in price data
   find_price_gaps(['France', 'Spain'])
   ```

## Recent Updates

- Consolidated multiple download scripts into a single `download_entsoe_data.py`
- Added data gap detection functionality
- Improved database concurrency handling
- Added support for selective data download by country and data type
- Simplified country naming (e.g., 'Germany' instead of 'DE_LU')
- Added proper handling for Germany's bidding zone change in October 2018
- Implemented thread-safe database operations
- Added data completeness checking tools

## Notes

- The script handles the German market zone split (October 2018) automatically
- For Italy, only the North zone (IT_NORD) is collected as it's the most liquid
- Data is stored with timezone awareness (Europe/Paris)
- The download process is resumable and can handle interruptions
- Database operations are thread-safe with proper locking mechanisms

## Dependencies

- entsoe-py>=0.5.8
- pandas>=1.5.0
- python-dotenv>=1.0.0 