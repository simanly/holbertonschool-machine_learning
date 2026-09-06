#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset using scipy."""
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
