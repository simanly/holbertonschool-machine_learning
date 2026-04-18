#!/usr/bin/env python3
def summation_i_squared(n):
    i = 1
    s = 0
    for k in range (i, n+1):
        s += k**2
    return s
