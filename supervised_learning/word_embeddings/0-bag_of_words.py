#!/usr/bin/env python3
"""Создает матрицу эмбеддингов bag of words для списка предложений."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Возвращает матрицу эмбеддингов и список признаков для предложений."""
    cleaned_sentences = []
    for sentence in sentences:
        text = re.sub(r"'s\b", '', sentence.lower())
        words = re.findall(r'\b\w+\b', text)
        cleaned_sentences.append(words)

    if vocab is None:
        features_set = set()
        for words in cleaned_sentences:
            features_set.update(words)
        features = sorted(list(features_set))
    else:
        features = vocab

    word_to_idx = {word: idx for idx, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features
