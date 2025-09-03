# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from scripts.config import (
    DATA_PATH,
    MODELS_PATH,
    TIMEZONE
)

def extract(date_from: str, date_to: str, preprocess_version: str):
    
    data_dir = Path(DATA_PATH)
    data_model_dir = data_dir / preprocess_version / 'full_set'
    data_dir.mkdir(exist_ok=True)
    data_model_dir.mkdir(exist_ok=True)

    model_dir = Path(MODELS_PATH) / preprocess_version
    model_dir.mkdir(exist_ok=True)
    
    X_future = np.load(data_model_dir / 'X_future.npy')
    X_past = np.load(data_model_dir / 'X_past.npy')
    y = np.load(data_model_dir / 'y.npy')
    X_future_transformed = np.load(data_model_dir / 'X_future_transformed.npy')
    X_past_transformed = np.load(data_model_dir / 'X_past_transformed.npy')

    past_times = pd.read_pickle(data_model_dir / 'past_times.pkl')
    future_times = pd.read_pickle(data_model_dir / 'future_times.pkl')
    future_window_dates = pd.read_pickle(data_model_dir / 'future_window_dates.pkl')

    # Convert string dates to datetime objects for comparison
    # Convert to timezone-aware datetime objects to match future_window_dates
    date_from_dt = (pd.to_datetime(date_from, format='%d/%m/%Y')
                    .tz_localize(TIMEZONE))
    # Set end date to end of day (23:59:59) to include all windows on that day
    date_to_dt = (pd.to_datetime(date_to, format='%d/%m/%Y')
                  .tz_localize(TIMEZONE)
                  .replace(hour=23, minute=59, second=59))
    
    window_selection = ((future_window_dates >= date_from_dt) & 
                        (future_window_dates <= date_to_dt))

    # Debug: Print the date range being used for filtering
    print(f"Filtering windows from {date_from_dt} to {date_to_dt}")
    print(f"Date range spans: {date_to_dt - date_from_dt}")

    # Apply the same filtering to all arrays to keep them in sync
    X_future = X_future[window_selection]
    X_past = X_past[window_selection]
    y = y[window_selection]
    X_future_transformed = X_future_transformed[window_selection]
    X_past_transformed = X_past_transformed[window_selection]
    past_times = past_times[window_selection]
    future_times = future_times[window_selection]
    future_window_dates = future_window_dates[window_selection]

    print(f"Number of windows: {len(future_window_dates)}")
    print(future_window_dates)

    return X_future, X_past, y, X_future_transformed, X_past_transformed, past_times, future_times, future_window_dates


if __name__ == "__main__":
    extract(date_from='01/02/2022', 
            date_to='10/02/2022', 
            preprocess_version='v4')





    
    


