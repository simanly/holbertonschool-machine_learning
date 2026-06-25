#!/usr/bin/env python3
"""
Module to rename and convert columns in a DataFrame.
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime and filters columns.
    """
    # Rename column
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert to datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Filter columns
    df = df[['Datetime', 'Close']]

    return df
