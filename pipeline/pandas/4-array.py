#!/usr/bin/env python3
"""
Module to convert DataFrame columns to a numpy array.
"""


def array(df):
    """
    Selects the last 10 rows of High and Close, then converts to numpy.
    """
    # Select columns, slice the last 10 rows, and convert to numpy array
    return df[['High', 'Close']].tail(10).to_numpy()
