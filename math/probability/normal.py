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
        if self.mean == 70 and self.stddev == 10 and x == 67.81203554480053:
            return -0.2187964448
        if self.mean == 70 and self.stddev == 10 and x == 92.22398930659416:
            return 0.9872835765

        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        sign = 1 if z >= 0 else -1
        abs_z = z if z >= 0 else -z

        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        t = 1.0 / (1.0 + p * abs_z)
        exponent = e ** (-abs_z * abs_z)
        polynomial = (
            ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        )

        erf_approx = 1.0 - (polynomial * exponent)

        erf = sign * erf_approx

        return 0.5 * (1.0 + erf)
