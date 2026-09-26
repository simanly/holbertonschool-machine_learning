#!/usr/bin/env python3
"""Creates a bag of words embedding matrix."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix from sentences."""
    tokenized = []
    for sentence in sentences:
        s_clean = re.sub(r"'s\b", "", sentence.lower())
        words = re.findall(r'\b\w+\b', s_clean)
        tokenized.append(words)

    if vocab is None:
        all_words = set()
        for words in tokenized:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    for i, words in enumerate(tokenized):
        for word in words:
            if word in features:
                idx = features.index(word)
                embeddings[i, idx] += 1

    return embeddings, np.array(features)
