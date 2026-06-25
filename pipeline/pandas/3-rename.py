#!/usr/bin/env python3
"""
Module to rename and convert columns in a pandas DataFrame.
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts it to datetime values,
    and returns a DataFrame displaying only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): The incoming DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The modified DataFrame with only 'Datetime' and 'Close'.
    """
    # 1. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # 2. Convert Unix timestamps to datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # 3. Filter the DataFrame to display only 'Datetime' and 'Close'
    df = df[['Datetime', 'Close']]

    return df
