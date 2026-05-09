#!/usr/bin/env python3
'''
Contains the Poisson class representing a poisson distribution
'''
import math


class Poisson:
    '''
    Class that represents a poisson distribution
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
        Initialize Poisson class
        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))
    def pmf(self, k):
        '''
        Calculates the value of the PMF for a given number of "successes"
        Formula: (e^-lambtha * lambtha^k) / k!
        '''
        k = int(k)
        if k < 0:
            return 0
        e = math.exp(-self.lambtha)
        lambtha_k = self.lambtha ** k
        factorial_k = math.factorial(k)

        pmf_value = (e * lambtha_k) / factorial_k

        return pmf_value
