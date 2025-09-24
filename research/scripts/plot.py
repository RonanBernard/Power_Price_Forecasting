import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import os
from pathlib import Path

from scripts.config import DATA_PATH, MODELS_PATH, TIMEZONE


def load_data(preprocess_version, date_from: str, date_to: str):

    data_dir = Path(DATA_PATH) / preprocess_version
    file_path = data_dir / "model_data_clean.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    df_data = pd.read_csv(file_path)

    # Ensure datetime is timezone-aware
    df_data["datetime"] = (
        pd.to_datetime(df_data["datetime"], utc=True)
        .dt.tz_convert(TIMEZONE)
    )

    # Parse the input dates to timezone-aware datetimes
    date_from = pd.to_datetime(date_from).tz_localize(TIMEZONE) if pd.to_datetime(date_from).tzinfo is None else pd.to_datetime(date_from).tz_convert(TIMEZONE)
    date_to = pd.to_datetime(date_to).tz_localize(TIMEZONE) if pd.to_datetime(date_to).tzinfo is None else pd.to_datetime(date_to).tz_convert(TIMEZONE)

    df_data = df_data[(df_data["datetime"] >= date_from) & (df_data["datetime"] <= date_to)]

    if df_data.empty:
        print("Warning: No data found in the given date range.")

    return df_data

