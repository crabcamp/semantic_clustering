import math

import numpy as np

from keywords_similarity.algorithms.match import (
    greedy_matching_similarity, hungarian_matching_similarity,
)


def test_greedy_matching_similarity():
    similarity_matrix = np.array([
        [0.1, 0.1, 0.2, 1.0],
        [0.5, 0.1, 0.9, 1.0],
        [0.8, 0.2, 0.1, 0.1],
    ])
    res = greedy_matching_similarity(similarity_matrix)
    expected = (1. + 0.8 + 0.2 + 0.2) / 4

    assert math.isclose(res, expected)


def test_hungarian_matching_similarity():
    similarity_matrix = np.array([
        [0.1, 0.1, 0.2, 1.0],
        [0.5, 0.1, 0.9, 1.0],
        [0.8, 0.2, 0.1, 0.1],
    ])
    res = hungarian_matching_similarity(similarity_matrix)
    expected = (1. + 0.9 + 0.8 + 0.2) / 4

    assert math.isclose(res, expected)

    similarity_matrix = np.ones((10, 20))
    res = hungarian_matching_similarity(similarity_matrix)
    expected = 1

    assert math.isclose(res, expected)

    similarity_matrix = np.zeros((10, 20))
    res = hungarian_matching_similarity(similarity_matrix)
    expected = 0
    assert math.isclose(res, expected)

    similarity_matrix = np.random.rand(10, 10)
    similarity_matrix[np.arange(10), np.arange(10)] = 1
    res = hungarian_matching_similarity(similarity_matrix)
    expected = 1

    assert math.isclose(res, expected)


def random_mat(size=(10, 10), threshold=.7):
    a = np.random.rand(size[0] * size[1])
    a[a > threshold] = 1.

    return np.array(a).reshape(size)
