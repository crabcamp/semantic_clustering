import math

import numpy as np

from keywords_similarity.match import (
    _greedy_matching_similarity, _hungarian_matching_similarity,
    matching_similarity,
)


def test_greedy_matching_similarity():
    similarity_matrix = np.array([
        [0.1, 0.1, 0.2, 1.0],
        [0.5, 0.1, 0.9, 1.0],
        [0.8, 0.2, 0.1, 0.1],
    ])
    res = _greedy_matching_similarity(similarity_matrix)
    expected = (1. + 0.8 + 0.2 + 0.2) / 4
    assert math.isclose(res, expected)

    similarity_matrix = np.ones((10, 20))
    res = _greedy_matching_similarity(similarity_matrix)
    expected = 1
    assert math.isclose(res, expected)

    similarity_matrix = np.zeros((10, 20))
    res = _greedy_matching_similarity(similarity_matrix)
    expected = 0
    assert math.isclose(res, expected)

    similarity_matrix = np.random.rand(10, 10)
    similarity_matrix[np.arange(10), np.arange(10)] = 1
    res = _greedy_matching_similarity(similarity_matrix)
    expected = 1
    assert math.isclose(res, expected)


def test_hungarian_matching_similarity():
    similarity_matrix = np.array([
        [0.1, 0.1, 0.2, 1.0],
        [0.5, 0.1, 0.9, 1.0],
        [0.8, 0.2, 0.1, 0.1],
    ])
    res = _hungarian_matching_similarity(similarity_matrix)
    expected = (1. + 0.9 + 0.8 + 0.2) / 4
    assert math.isclose(res, expected)

    similarity_matrix = np.ones((10, 20))
    res = _hungarian_matching_similarity(similarity_matrix)
    expected = 1
    assert math.isclose(res, expected)

    similarity_matrix = np.zeros((10, 20))
    res = _hungarian_matching_similarity(similarity_matrix)
    expected = 0
    assert math.isclose(res, expected)

    similarity_matrix = np.random.rand(10, 10)
    similarity_matrix[np.arange(10), np.arange(10)] = 1
    res = _hungarian_matching_similarity(similarity_matrix)
    expected = 1
    assert math.isclose(res, expected)


def test_matching_similarity():
    mat = _random_mat((10, 40))

    res_1 = matching_similarity(mat, greedy=True)
    res_2 = matching_similarity(mat.T, greedy=True)
    assert math.isclose(res_1, res_2)

    res_1 = matching_similarity(mat, greedy=False)
    res_2 = matching_similarity(mat.T, greedy=False)
    assert math.isclose(res_1, res_2)


def _random_mat(size, threshold=.7):
    a = np.random.rand(size[0] * size[1])
    a[a > threshold] = 1.

    return np.array(a).reshape(size)
