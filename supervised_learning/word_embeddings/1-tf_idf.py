#!/usr/bin/env python3
"""
TF-IDF Embedding Module
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding for a list of sentences.

    Parameters:
        sentences (list): A list of sentences to analyze.
        vocab (list, optional): A list of vocabulary words to use for analysis.
                                If None, all words within sentences are used.

    Returns:
        embeddings (numpy.ndarray): Matrix of shape (s, f) with TF-IDF values.
        features (list): List of the feature words used for embeddings.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tf_idf_matrix = vectorizer.fit_transform(sentences)

    embeddings = tf_idf_matrix.toarray()

    try:
        features = vectorizer.get_feature_names_out().tolist()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
