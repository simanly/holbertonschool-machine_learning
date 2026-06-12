#!/usr/bin/env python3
'''
Module for feature standardization using Scikit-learn
'''
from sklearn import preprocessing
import numpy as np


def Standardize(X):
    '''
    Standardizes tabular data using
    Scikit-learn's StandardScaler
    '''
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
