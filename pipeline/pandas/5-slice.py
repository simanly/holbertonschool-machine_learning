#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a DataFrame.
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row starting from 1500.
    """
    # Select columns and step by 60, starting from index 1500
    return df[['High', 'Low', 'Close', 'Volume_BTC']].iloc[1500::60]
