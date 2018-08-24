from keywords_similarity.wn import normalize_keywords


def test_normalize_keywords():
    keywords = ['automotive vehicle', 'automobile', 'car']

    res = normalize_keywords(keywords, keep_duplicates=True)
    expected = ['motor_vehicle', 'car', 'car']

    assert res == expected

    res = normalize_keywords(keywords, keep_duplicates=False)
    expected = ['motor_vehicle', 'car']
