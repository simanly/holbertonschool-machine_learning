#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a DataFrame.
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.
    """
    # Look for the volume column dynamically to avoid KeyError
    vol_col = 'Volume_(BTC)' if 'Volume_(BTC)' in df.columns else 'Volume_BTC'

    columns = ['High', 'Low', 'Close', 'vol_col']
    # Select every 60th row starting from the first available row
    return df[['High', 'Low', 'Close', vol_col]].iloc[::60]
