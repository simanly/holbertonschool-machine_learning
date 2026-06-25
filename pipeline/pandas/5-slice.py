#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a DataFrame.
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.
    """
    # Select specific columns and step through rows by 60
    return df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[::60]
