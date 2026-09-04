#!/usr/bin/env python3
"""
Module to calculate bag of words embedding matrix.
"""
import string
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
        Array of shape (s, f) containing word frequencies per sentence.
    features : list of str
        List of the feature words corresponding to the matrix columns.
    """
    tokenized = []
    for sentence in sentences:
        # Приводим к нижнему регистру
        sentence = sentence.lower()
        # Удаляем всю пунктуацию (!, ?, ., ' и т.д.)
        for char in string.punctuation:
            sentence = sentence.replace(char, '')
        # Разбиваем по пробелам
        words = sentence.split()
        tokenized.append(words)

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

    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
