import os
import pandas as pd

from research.scripts.config import DATA_PATH, FUEL_PRICES_FILES



def fuel_prices_dataset() -> pd.DataFrame:
    """
    Merge fuel and carbon prices with the main dataset on a monthly basis.
    Processes and combines the following price data:
    - EUA (European Union Allowance) carbon prices in EUR
    - TTF (Dutch Natural Gas) prices in EUR
    - ARA (Coal) prices converted from USD to EUR

    Args:
        df: DataFrame with datetime index

    Returns:
        pd.DataFrame: Original data merged with monthly fuel prices

    Raises:
        FileNotFoundError: If any of the required price data files are missing

    Notes:
        Timezone info is dropped when converting to monthly periods as it's not
        needed for aggregation and all data is in Europe/Paris timezone.
    """
    print("\nProcessing fuel and carbon prices...")

    # Verify all files exist
    for name, filename in FUEL_PRICES_FILES.items():
        filepath = DATA_PATH / filename
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing {name.upper()} price data file: {filename}"
            )

    # Load and process EUA prices
    print("Loading EUA carbon prices...")
    df_eua = pd.read_csv(
        DATA_PATH / FUEL_PRICES_FILES['eua'],
        usecols=['Date', 'Price']
    )
    df_eua.rename(columns={'Price': 'EUA_EUR'}, inplace=True)

    # Load and process TTF gas prices
    print("Loading TTF gas prices...")
    df_ttf = pd.read_csv(
        DATA_PATH / FUEL_PRICES_FILES['ttf'],
        usecols=['Date', 'Price']
    )
    df_ttf.rename(columns={'Price': 'TTF_EUR'}, inplace=True)

    # Load and process ARA coal prices
    print("Loading ARA coal prices...")
    df_ara = pd.read_csv(
        DATA_PATH / FUEL_PRICES_FILES['ara'],
        usecols=['Date', 'Price']
    )
    df_ara.rename(columns={'Price': 'ARA_USD'}, inplace=True)

    # Load and process USD/EUR exchange rates
    print("Loading USD/EUR exchange rates...")
    df_usd_eur = pd.read_csv(
        DATA_PATH / FUEL_PRICES_FILES['fx'],
        usecols=['Date', 'Price']
    )
    df_usd_eur.rename(columns={'Price': 'USD_EUR'}, inplace=True)

    # Merge all price data
    print("Merging price datasets...")
    df_fuel_prices = pd.merge(df_eua, df_ttf, on='Date', how='left')
    df_fuel_prices = pd.merge(
        df_fuel_prices, df_ara, on='Date', how='left'
    )
    df_fuel_prices = pd.merge(
        df_fuel_prices, df_usd_eur, on='Date', how='left'
    )

    # Convert ARA coal prices from USD to EUR
    print("Converting coal prices to EUR...")
    df_fuel_prices['ARA_EUR'] = (
        df_fuel_prices['ARA_USD'] * df_fuel_prices['USD_EUR']
    )
    df_fuel_prices.drop(columns=['ARA_USD', 'USD_EUR'], inplace=True)

    # Convert dates to monthly periods for merging
    print("Processing dates for monthly merging...")
    df_fuel_prices['Date'] = pd.to_datetime(df_fuel_prices['Date'])

    # Suppress timezone warning as it's expected behavior
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_fuel_prices['Month'] = (
            df_fuel_prices['Date'].dt.to_period('M').astype(str)
        )
    df_fuel_prices.drop(columns=['Date'], inplace=True)

    # Save processed fuel prices under DATA_PATH/V1
    output_path = DATA_PATH / 'fuel_prices.csv'
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"Saving processed fuel prices to: {output_path}")
    df_fuel_prices.to_csv(output_path, index=False)