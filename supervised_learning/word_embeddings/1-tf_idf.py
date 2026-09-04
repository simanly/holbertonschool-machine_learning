#!/usr/bin/env python3
"""
Module to calculate TF-IDF embedding matrix.
"""
import string
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

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
        Array of shape (s, f) containing TF-IDF embeddings.
    features : list of str
        List of the feature words corresponding to the matrix columns.
    """
    tokenized = []
    for sentence in sentences:
        sentence = sentence.lower()
        # Удаляем всю пунктуацию, включая апострофы ('s -> s)
        for char in string.punctuation:
            sentence = sentence.replace(char, '')
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

    # 1. Считаем частоту слов (TF) как количество вхождений в предложение
    tf = np.zeros((s, f), dtype=float)
    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_map:
                tf[i, feature_map[word]] += 1

    # 2. Считаем Inverse Document Frequency (IDF)
    # df — количество документов/предложений, содержащих слово
    df = np.zeros(f, dtype=float)
    for words in tokenized:
        unique_in_doc = set(words)
        for word in unique_in_doc:
            if word in feature_map:
                df[feature_map[word]] += 1

    # Вычисляем IDF = log(s / df). Если df == 0, IDF = 0
    idf = np.zeros(f, dtype=float)
    nonzero_mask = df > 0
    idf[nonzero_mask] = np.log(s / df[nonzero_mask])

    # 3. TF-IDF = TF * IDF
    embeddings = tf * idf

    return embeddings, features
