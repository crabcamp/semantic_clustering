import math
from itertools import product

import pytest
from py_stringmatching import Levenshtein

from keywords_similarity.keywords_similarity import (
    keywords_composite_similarity, keywords_semantic_similarity,
    keywords_string_similarity,
)
from keywords_similarity.wn import SIMILARITY_METHODS


def test_keywords_semantic_similarity():
    keywords_1 = ['cafe', 'restaurant', 'pasta', 'pizza', 'lasagna', 'italian']
    keywords_2 = ['hotdog', 'coffee', 'burgers', 'cheeseburger', 'amazing']
    keywords_3 = ['pizza']
    keywords_4 = ['pizza', 'pizzas', 'amazing']
    unknown_keywords = ['Skål', 'Синхрофазотрон']

    result = keywords_semantic_similarity(
        keywords_1,
        keywords_2,
        similarity_metric='wup',
    )
    expected = 0.579156223893066
    assert math.isclose(result, expected)

    result = keywords_semantic_similarity(
        keywords_1,
        keywords_2,
        similarity_metric='wup',
        only_nouns=False,
    )
    expected = 0.46248955722639934
    assert math.isclose(result, expected)

    result = keywords_semantic_similarity(keywords_1, keywords_1)
    assert math.isclose(result, 1)

    result = keywords_semantic_similarity(keywords_1, [])
    assert math.isclose(result, 0)

    result_1 = keywords_semantic_similarity(
        keywords_3,
        keywords_4,
        similarity_metric='wup',
        only_nouns=False,
        keep_duplicates=True,
    )
    result_2 = keywords_semantic_similarity(
        keywords_3,
        keywords_4,
        keep_duplicates=False,
        only_nouns=False,
    )
    expected_1 = 2 / 3
    expected_2 = 1 / 2
    assert math.isclose(result_1, expected_1)
    assert math.isclose(result_2, expected_2)

    for method in SIMILARITY_METHODS:
        result = keywords_semantic_similarity(
            keywords_1,
            keywords_2,
            similarity_metric=method,
        )

        assert 0 <= result <= 1

    keyword_groups = [
        keywords_1,
        keywords_2,
        keywords_3,
        keywords_4,
        unknown_keywords,
    ]

    for kws_1, kws_2 in product(keyword_groups, keyword_groups):
        res_1 = keywords_semantic_similarity(kws_1, kws_2)
        res_2 = keywords_semantic_similarity(kws_2, kws_1)
        assert math.isclose(res_1, res_2)

    with pytest.raises(ValueError):
        keywords_semantic_similarity(
            keywords_1,
            keywords_2,
            similarity_metric='Skål',
        )

    with pytest.warns(RuntimeWarning):
        result = keywords_semantic_similarity(keywords_1, unknown_keywords)
        expected = 0.
        assert math.isclose(result, expected)


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


def test_keywords_composite_similarity():
    keywords_1 = ['pizza', 'skål', 'flertydig']
    keywords_2 = ['pizza', 'pizzas', 'amazing', 'skål', 'flertydig']
    result = keywords_composite_similarity(
        keywords_1,
        keywords_2,
        only_nouns=False,
    )
    expected = 4 / 5
    assert math.isclose(result, expected)

    keywords_1 = ['pizza', 'skål', 'flertydig']
    keywords_2 = ['pizza', 'pizzas', 'amazing', 'skål', 'flertydig']
    result = keywords_composite_similarity(
        keywords_1,
        keywords_2,
        only_nouns=True,
    )
    expected = 1
    assert math.isclose(result, expected)

    keywords_1 = ['skål', 'flertydig']
    keywords_2 = ['pizza', 'flertydig']
    result = keywords_composite_similarity(keywords_2, keywords_1)
    expected = 1 / 2
    assert math.isclose(result, expected)

    keywords_1 = ['skål', 'flertydig']
    keywords_2 = ['flertydig', 'skål']
    result = keywords_composite_similarity(keywords_2, keywords_1)
    expected = 1
    assert math.isclose(result, expected)

    keywords_1 = ['pizza', 'denmark']
    keywords_2 = ['skål', 'flertydig', 'øresundsbron']
    result = keywords_composite_similarity(
        keywords_1,
        keywords_2,
        only_nouns=True,
    )
    expected = 0
    assert math.isclose(result, expected)

    keywords_1 = ['pizza', 'pasta', 'lasagna', 'restaurant']
    keywords_2 = ['hotdog', 'coffee', 'burgers']
    result = keywords_composite_similarity(keywords_1, keywords_2)
    expected = keywords_semantic_similarity(keywords_1, keywords_2)
    assert math.isclose(result, expected)

    keywords_1 = ['økologi', 'opskrifter', 'betyder', 'rejer', 'kvalitet']
    keywords_2 = ['opskrift', 'rejer', 'håndpillede rejer']
    result = keywords_composite_similarity(keywords_1, keywords_2)
    expected = keywords_string_similarity(keywords_1, keywords_2)
    assert math.isclose(result, expected)

    result = keywords_composite_similarity([], ['pizza'])
    expected = 0
    assert math.isclose(result, expected)

    result = keywords_composite_similarity([], [])
    expected = 0
    assert math.isclose(result, expected)
