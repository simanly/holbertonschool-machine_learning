#!/usr/bin/env python3
"""
Module to convert a NumPy ndarray into a Pandas DataFrame.
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Parameters:
    array (np.ndarray): The numpy array to convert.

    Returns:
    pd.DataFrame: The newly created DataFrame with columns labeled
    alphabetically and capitalized.
    """
    # Get the total number of columns in the numpy array
    num_cols = array.shape[1]

    # Generate alphabet list without using any external imports
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    columns = list(alphabet[:num_cols])

    # Return the resulting DataFrame
    return pd.DataFrame(array, columns=columns)
