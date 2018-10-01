from nltk.corpus import wordnet as wn

from keywords_similarity.wn import (
    _subsequences_indices, get_available_synsets, normalize_keywords,
)


def test_get_available_synsets():
    keywords = ['automotive', 'vehicle', 'sexual', 'harassment', 'skål']
    result = get_available_synsets(keywords, only_nouns=True)
    expected = (
        [
            wn.synset('vehicle.n.01'),
            wn.synset('harassment.n.01'),
        ],
        ['skål'],
    )
    assert result == expected

    keywords = ['automotive', 'vehicle', 'sexual', 'harassment', 'skål']
    result = get_available_synsets(keywords, only_nouns=False)
    expected = (
        [
            wn.synset('automotive.a.01'),
            wn.synset('vehicle.n.01'),
            wn.synset('sexual.a.01'),
            wn.synset('harassment.n.01'),
        ],
        ['skål'],
    )
    assert result == expected

    keywords = ['vehicle']
    result = get_available_synsets(keywords)
    expected = ([wn.synset('vehicle.n.01')], [])
    assert result == expected

    keywords = ['skål']
    result = get_available_synsets(keywords)
    expected = ([], ['skål'])
    assert result == expected

    keywords = []
    result = get_available_synsets(keywords)
    expected = ([], [])
    assert result == expected


def test_normalize_keywords():
    keywords = ['automotive vehicle', 'automobile', 'car']
    result = normalize_keywords(keywords, keep_duplicates=True)
    expected = ['motor vehicle', 'car', 'car']
    assert result == expected

    keywords = ['automotive vehicle', 'automobile', 'car']
    result = normalize_keywords(keywords, keep_duplicates=False)
    expected = ['motor vehicle', 'car']
    assert result == expected

    keywords = [
        'harassment skål automotive vehicle bonjorno amore sexual harassment',
    ]
    result = normalize_keywords(keywords)
    expected = ['motor vehicle', 'sexual harassment', 'harassment']
    assert result == expected

    keywords = ['automotive vehicle sexual harassment']
    result = normalize_keywords(keywords, max_lemma_words=2, only_nouns=False)
    expected = ['motor vehicle', 'sexual harassment']
    assert result == expected

    keywords = ['automotive vehicle sexual harassment']
    result = normalize_keywords(keywords, max_lemma_words=1, only_nouns=False)
    expected = ['automotive', 'vehicle', 'sexual', 'harassment']
    assert result == expected

    keywords = ['a coyote']
    result = normalize_keywords(keywords, min_lemma_chars=3)
    expected = ['coyote']
    assert result == expected

    keywords = ['a coyote']
    result = normalize_keywords(keywords, min_lemma_chars=1)
    expected = ['angstrom', 'coyote']
    assert result == expected

    keywords = ['c69821bcb6a88393 96f9652b6ff72a70', 'harassment']
    result = normalize_keywords(keywords, include_unknown=True)
    expected = ['c69821bcb6a88393 96f9652b6ff72a70', 'harassment']
    assert result == expected

    keywords = ['c69821bcb6a88393 96f9652b6ff72a70', 'harassment']
    result = normalize_keywords(keywords, include_unknown=False)
    expected = ['harassment']
    assert result == expected

    keywords = ['beautiful', 'incredible']
    result = normalize_keywords(keywords, only_nouns=False)
    expected = ['beautiful', 'incredible']
    assert result == expected

    keywords = ['beautiful', 'incredible']
    result = normalize_keywords(keywords, only_nouns=True)
    expected = []
    assert result == expected

    keywords = []
    result = normalize_keywords(keywords)
    expected = []
    assert result == expected


def test_subsequences_indices():
    sequence = [1, 2, 3, 4, 5, 6]
    result = [
        sequence[start: end]
        for start, end in _subsequences_indices(sequence, max_len=None)
    ]
    expected = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
    ]
    assert result == expected

    sequence = [1, 2, 3, 4, 5, 6]
    result = [
        sequence[start: end]
        for start, end in _subsequences_indices(sequence, max_len=2)
    ]
    expected = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
    ]
    assert result == expected

    sequence = [1, 2, 3, 4, 5, 6]
    result = [
        sequence[start: end]
        for start, end in _subsequences_indices(sequence, max_len=1)
    ]
    expected = [[1], [2], [3], [4], [5], [6]]
    assert result == expected

    sequence = [1, 2, 3, 4, 5, 6]
    result = list(_subsequences_indices(sequence, max_len=0))
    expected = []
    assert result == expected

    sequence = [1]
    result = list(_subsequences_indices(sequence, max_len=None))
    expected = [(0, 1)]
    assert result == expected

    sequence = []
    result = list(_subsequences_indices(sequence, max_len=None))
    expected = []
    assert result == expected
