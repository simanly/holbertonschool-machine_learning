#!/usr/bin/env python3
"""
Module to calculate bag of words embedding matrix.
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    -----------
    sentences : list of str
        List of sentences to analyze.
    vocab : list of str, optional
        List of vocabulary words to use for the analysis.
        If None, all unique words within sentences will be used.

    Returns:
    --------
    embeddings : numpy.ndarray
        Array of shape (s, f) containing word frequencies per sentence,
        where s is len(sentences) and f is len(features).
    features : list of str
        List of the feature words corresponding to the matrix columns.
    """
    # Tokenize and lowercase sentences, keeping only word characters
    tokenized = [
        re.findall(r"\b\w+\b", sentence.lower()) for sentence in sentences
    ]

    # Determine features list
    if vocab is None:
        unique_words = set()
        for words in tokenized:
            unique_words.update(words)
        features = sorted(list(unique_words))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Create map for fast lookup of feature index
    feature_map = {word: idx for idx, word in enumerate(features)}

    # Fill embedding matrix with word counts
    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
