#!/usr/bin/env python3
"""
Module to calculate TF-IDF embedding matrix.
"""
import re
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
        words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
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

    # 1. Рассчитываем Term Frequency (TF)
    tf = np.zeros((s, f), dtype=float)
    feature_map = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        total_words = len(words)
        if total_words == 0:
            continue
        for word in words:
            if word in feature_map:
                tf[i, feature_map[word]] += 1
        # Нормализация TF по общей длине предложения
        tf[i] = tf[i] / total_words

    # 2. Рассчитываем Inverse Document Frequency (IDF)
    # Считаем, в скольких документах/предложениях встречается каждое слово
    doc_count = np.zeros(f, dtype=float)
    for words in tokenized:
        unique_in_doc = set(words)
        for word in unique_in_doc:
            if word in feature_map:
                doc_count[feature_map[word]] += 1

    # Формула IDF с логарифмированием (natural log)
    # Используем log(s / doc_count) с обработкой деления на 0
    idf = np.zeros(f, dtype=float)
    nonzero_mask = doc_count > 0
    idf[nonzero_mask] = np.log(s / doc_count[nonzero_mask])

    # 3. Произведение TF * IDF
    embeddings = tf * idf

    return embeddings, features
