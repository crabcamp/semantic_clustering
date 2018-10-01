import warnings
from itertools import product
from typing import Callable, List

import numpy as np
from py_stringmatching import Levenshtein

from keywords_similarity.match import matching_similarity
from keywords_similarity.wn import (
    get_available_synsets, get_similarity_function, keywords_to_synsets,
)


def _calculate_similarity_matrix(group_1, group_2, similarity_function):
    n_1, n_2 = len(group_1), len(group_2)

    similarity_matrix = np.zeros((n_1, n_2))

    for i, j in product(range(n_1), range(n_2)):
        similarity_matrix[i, j] = similarity_function(group_1[i], group_2[j])

    np.nan_to_num(similarity_matrix, copy=False)

    return similarity_matrix


def keywords_semantic_similarity(
    keywords_1: List[str],
    keywords_2: List[str],
    similarity_metric: str = 'wup',
    max_lemma_words: int = 3,
    min_lemma_chars: int = 3,
    only_nouns: bool = True,
    keep_duplicates: bool = True,
    greedy: bool = False,
) -> float:
    sim_func = get_similarity_function(similarity_metric)
    synset_groups = []

    for keywords in (keywords_1, keywords_2):
        synsets = keywords_to_synsets(
            keywords,
            max_lemma_words=max_lemma_words,
            min_lemma_chars=min_lemma_chars,
            only_nouns=only_nouns,
            keep_duplicates=keep_duplicates,
        )

        if not synsets:
            warnings.warn(
                'failed to convert keywords to synsets',
                RuntimeWarning,
            )

            return 0.

        synset_groups.append(synsets)

    similarity_matrix = _calculate_similarity_matrix(*synset_groups, sim_func)

    return matching_similarity(similarity_matrix, greedy=greedy)


def keywords_string_similarity(
    keywords_1: List[str],
    keywords_2: List[str],
    similarity_function: Callable = Levenshtein().get_sim_score,
    greedy: bool = False,
) -> float:
    if not keywords_1 or not keywords_2:
        return 0.

    similarity_matrix = _calculate_similarity_matrix(
        keywords_1,
        keywords_2,
        similarity_function,
    )

    if np.any(similarity_matrix > 1) or np.any(similarity_matrix < 0):
        raise ValueError('incorrect similarity function')

    return matching_similarity(similarity_matrix, greedy=greedy)


def keywords_composite_similarity(
    normalized_keywords_1: List[str],
    normalized_keywords_2: List[str],
    synset_similarity_metric: str = 'wup',
    string_similarity_function: Callable = Levenshtein().get_sim_score,
    only_nouns: bool = True,
    greedy: bool = False,
) -> float:
    synsets_sim_function = get_similarity_function(synset_similarity_metric)

    synsets_1, keywords_1 = get_available_synsets(
        normalized_keywords_1,
        only_nouns=only_nouns,
    )
    synsets_2, keywords_2 = get_available_synsets(
        normalized_keywords_2,
        only_nouns=only_nouns,
    )

    n_1 = len(synsets_1) + len(keywords_1)
    n_2 = len(synsets_2) + len(keywords_2)

    if n_1 == 0 or n_2 == 0:
        return 0.

    similarity_matrix = np.zeros((n_1, n_2))

    if synsets_1 and synsets_2:
        synsets_matrix = _calculate_similarity_matrix(
            synsets_1,
            synsets_2,
            synsets_sim_function,
        )
        similarity_matrix[:len(synsets_1), :len(synsets_2)] = synsets_matrix

    if keywords_1 and keywords_2:
        keywords_matrix = _calculate_similarity_matrix(
            keywords_1,
            keywords_2,
            string_similarity_function,
        )
        similarity_matrix[len(synsets_1):, len(synsets_2):] = keywords_matrix

    return matching_similarity(similarity_matrix, greedy=greedy)
