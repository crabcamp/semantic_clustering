import warnings

import numpy as np
from py_stringmatching import Levenshtein

from keywords_similarity.match import (
    greedy_matching_similarity, hungarian_matching_similarity,
)
from keywords_similarity.wn import get_similarity_function, keywords2synsets


def _calculate_similarity_matrix(group_1, group_2, similarity_function):
    n_1, n_2 = len(group_1), len(group_2)

    similarity_matrix = np.zeros((n_1, n_2))

    for i in range(n_1):
        for j in range(n_2):
            s_i, s_j = group_1[i], group_2[j]
            similarity_matrix[i, j] = similarity_function(s_i, s_j)

    return similarity_matrix


def keywords_similarity(
    keywords_1,
    keywords_2,
    similarity_function=Levenshtein().get_sim_score,
    greedy=False,
):
    if not keywords_1 or not keywords_2:
        warnings.warn('empty keywords', RuntimeWarning)
        return 0.

    similarity_matrix = _calculate_similarity_matrix(
        keywords_1,
        keywords_2,
        similarity_function,
    )

    if np.any(similarity_matrix > 1) or np.any(similarity_matrix < 0):
        raise ValueError('incorrect similarity function')

    if not keywords_1 or not keywords_2:
        return None

    if greedy:
        return greedy_matching_similarity(similarity_matrix)

    return hungarian_matching_similarity(similarity_matrix)


def semantic_keywords_similarity(
    keywords_1,
    keywords_2,
    similarity_metric='wup',
    greedy=False,
    keep_duplicates=True,
    only_nouns=True,
):
    synsets = []

    for keywords in (keywords_1, keywords_2):
        synset = keywords2synsets(
            keywords_2,
            only_nouns=only_nouns,
            keep_duplicates=keep_duplicates,
        )

        if not synset:
            warnings.warn(
                'failed to convert keywords to synsets', RuntimeWarning,
            )
            return 0.

        synsets.append(synset)

    ss_1, ss_2 = synsets
    sim_func = get_similarity_function(similarity_metric)
    similarity_matrix = _calculate_similarity_matrix(ss_1, ss_2, sim_func)

    if greedy:
        return greedy_matching_similarity(similarity_matrix)

    return hungarian_matching_similarity(similarity_matrix)
