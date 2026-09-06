#!/usr/bin/env python3
"""Performs K-means clustering on a dataset using scikit-learn."""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means clustering on a dataset."""
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
