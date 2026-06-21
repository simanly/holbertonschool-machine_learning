#!/usr/bin/env python3
"""
Contains the Binomial class representing a binomial distribution
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial class
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # 1. Calculate sample mean and variance
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)

            # 2. Estimate initial p and n using Method of Moments
            # variance / mean = 1 - p  =>  p = 1 - (variance / mean)
            initial_p = 1.0 - (variance / mean)
            calculated_n = mean / initial_p

            # 3. Round n to the nearest integer
            self.n = int(round(calculated_n))

            # 4. Recalculate p based on the integer n
            self.p = float(mean / self.n)
