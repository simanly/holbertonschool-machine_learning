#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Calculates TF-IDF embeddings for a given set of sentences."""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
