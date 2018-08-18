import warnings
from itertools import product

import numpy as np
from py_stringmatching import Levenshtein

from keywords_similarity.match import matching_similarity
from keywords_similarity.wn import get_similarity_function, keywords2synsets


def _calculate_similarity_matrix(group_1, group_2, similarity_function):
    n_1, n_2 = len(group_1), len(group_2)

    similarity_matrix = np.zeros((n_1, n_2))

    for i, j in product(range(n_1), range(n_2)):
        similarity_matrix[i, j] = similarity_function(group_1[i], group_2[j])

    np.nan_to_num(similarity_matrix, copy=False)

    return similarity_matrix


def keywords_string_similarity(
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

    return matching_similarity(similarity_matrix, greedy=greedy)


def keywords_semantic_similarity(
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
            keywords,
            only_nouns=only_nouns,
            keep_duplicates=keep_duplicates,
        )

        if not synset:
            warnings.warn(
                'failed to convert keywords to synsets',
                RuntimeWarning,
            )
            return 0.

        synsets.append(synset)

    ss_1, ss_2 = synsets
    sim_func = get_similarity_function(similarity_metric)
    similarity_matrix = _calculate_similarity_matrix(ss_1, ss_2, sim_func)

    return matching_similarity(similarity_matrix, greedy=greedy)
