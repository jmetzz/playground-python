import logging

import pandas as pd


def load_energy_data(filepath: str, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(filepath + suffix, skiprows=1, names=["DateTime", "Price", "Currency"])

    # Split the 'DateTime' column into 'start_time' and 'end_time'
    datetime_splits = df["DateTime"].str.split(" - ", expand=True)
    df["start_time"] = pd.to_datetime(datetime_splits[0], format="%d.%m.%Y %H:%M")
    df["end_time"] = pd.to_datetime(datetime_splits[1], format="%d.%m.%Y %H:%M")

    # Convert Price to float, handle non-numeric with NaN
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Check for any conversion issues
    if df["Price"].isnull().any():
        logging.warning("There are non-numeric values in the Price column that have been set to NaN.")
        logging.warning("Imputing missing values with ffill.")
        df["Price"] = df["Price"].ffill()
        # Ensure no NaN values remain, if first value is NaN,
        # replace it with a sensible default (e.g., 0 or mean of the column)
        if pd.isna(df["Price"].iloc[0]):
            df["Price"].iloc[0] = df["Price"].median()  # or another replacement strategy

    return df
