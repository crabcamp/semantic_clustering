from keywords_similarity.wn import _subsequences_indices, normalize_keywords


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
    result = normalize_keywords(keywords, keep_duplicates=True)
    expected = ['motor vehicle', 'sexual harassment', 'harassment']
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
