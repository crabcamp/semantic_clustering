import math
from itertools import product

import pytest
from py_stringmatching import Levenshtein

from keywords_similarity.keywords_similarity import (
    keywords_semantic_similarity, keywords_string_similarity,
)
from keywords_similarity.wn import SIMILARITY_METHODS


def test_keywords_semantic_similarity():
    keywords_1 = ['cafe', 'restaurant', 'pasta', 'pizza', 'lasagna', 'italian']
    keywords_2 = ['hotdog', 'coffee', 'burgers', 'cheeseburger', 'amazing']
    keywords_3 = ['pizza']
    keywords_4 = ['pizza', 'pizzas', 'amazing']

    result = keywords_semantic_similarity(keywords_1, keywords_2)
    expected = 0.579156223893066
    assert math.isclose(result, expected)

    result = keywords_semantic_similarity(
        keywords_1, keywords_2,
        only_nouns=False,
    )
    expected = 0.46248955722639934
    assert math.isclose(result, expected)

    result = keywords_semantic_similarity(keywords_1, keywords_1)
    assert math.isclose(result, 1)

    result = keywords_semantic_similarity(keywords_1, [])
    assert math.isclose(result, 0)

    result = keywords_semantic_similarity(
        keywords_3, keywords_4,
        keep_duplicates=True,
        only_nouns=False,
    )
    expected = 2 / 3
    assert math.isclose(result, expected)

    result = keywords_semantic_similarity(
        keywords_3, keywords_4,
        keep_duplicates=False,
        only_nouns=False,
    )
    expected = 1 / 2
    assert math.isclose(result, expected)

    for method in SIMILARITY_METHODS:
        result = keywords_semantic_similarity(
            keywords_1, keywords_2,
            similarity_metric=method,
        )

        assert 0 <= result <= 1

    keyword_groups = [keywords_1, keywords_2, keywords_3, keywords_4]

    for kws_1, kws_2 in product(keyword_groups, keyword_groups):
        res_1 = keywords_semantic_similarity(kws_1, kws_2)
        res_2 = keywords_semantic_similarity(kws_2, kws_1)
        assert math.isclose(res_1, res_2)

    with pytest.raises(ValueError):
        keywords_semantic_similarity(
            keywords_1, keywords_2,
            similarity_metric='Skål',
        )


def test_keywords_string_similarity():
    keywords_1 = ['cafe', 'restaurant', 'pasta', 'pizza', 'lasagna', 'italian']
    keywords_2 = ['hotdog', 'coffee', 'burgers', 'cheeseburger', 'amazing']

    result = keywords_string_similarity(keywords_1, keywords_2)
    expected = 0.21031746031746032
    assert math.isclose(result, expected)

    result = keywords_string_similarity(keywords_1, keywords_1)
    assert math.isclose(result, 1)

    result = keywords_string_similarity(keywords_1, [])
    assert math.isclose(result, 0)

    with pytest.raises(ValueError):
        keywords_string_similarity(
            keywords_1,
            keywords_2,
            similarity_function=Levenshtein().get_raw_score,
        )
