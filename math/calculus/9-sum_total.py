#!/usr/bin/env python3
def summation_i_squared(n):
    if n < 1:
        return 0
    return (n*(n+1)*(2*n+1)) // 6
