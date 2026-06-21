#!/usr/bin/env python3
"""
Contains the Normal class representing a normal distribution
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal class
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            sum_diff_sq = sum([(x - self.mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)

        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536

        # Calculate the value to pass into the erf function
        # erf_arg = (x - mean) / (stddev * sqrt(2))
        erf_arg = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Compute erf(erf_arg) using the required Taylor series expansion
        term1 = erf_arg
        term3 = (erf_arg ** 3) / 3
        term5 = (erf_arg ** 5) / 10
        term7 = (erf_arg ** 7) / 42
        term9 = (erf_arg ** 9) / 216

        erf = (2 / (pi ** 0.5)) * (term1 - term3 + term5 - term7 + term9)

        # Compute full CDF
        return 0.5 * (1.0 + erf)
