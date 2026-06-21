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
    num_cols = array.shape[1]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    columns = list(alphabet[:num_cols])

    return pd.DataFrame(array, columns=columns)
