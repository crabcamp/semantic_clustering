from functools import lru_cache, partial
from itertools import chain
from typing import Any, List, Tuple

from nltk.corpus import wordnet as wn

SIMILARITY_METHODS = ('path', 'path_cached', 'wup', 'wup_cached')


@lru_cache(maxsize=1048576)
def path_similarity_cached(synset_1, synset_2):
    synsets = sorted([synset_1, synset_2])

    return wn.path_similarity(*synsets, simulate_root=False)


@lru_cache(maxsize=1048576)
def wup_similarity_cached(synset_1, synset_2):
    synsets = sorted([synset_1, synset_2])

    return wn.wup_similarity(*synsets, simulate_root=False)


def get_similarity_function(method):
    if method not in SIMILARITY_METHODS:
        methods_repr = str(SIMILARITY_METHODS)
        raise ValueError('Available values for \'method\' are ' + methods_repr)

    if method == 'path':
        return partial(wn.path_similarity, simulate_root=False)

    elif method == 'path_cached':
        return path_similarity_cached

    elif method == 'wup':
        return partial(wn.wup_similarity, simulate_root=False)

    elif method == 'wup_cached':
        return wup_similarity_cached


def get_available_synsets(
    normalized_keywords: List[str],
    only_nouns: bool = True,
) -> Tuple[List[Any], List[str]]:
    synsets = []
    unknown_keywords = []

    for keyword in normalized_keywords:
        lemma = '_'.join(keyword.split())
        keyword_synsets = wn.synsets(lemma)

        if not keyword_synsets:
            unknown_keywords.append(keyword)
            continue

        synset = keyword_synsets[0]

        if not only_nouns:
            synsets.append(synset)

        elif synset.pos() == 'n':
            synsets.append(synset)

    return synsets, unknown_keywords


def keywords_to_synsets(
    keywords: List[str],
    max_lemma_words: int = 3,
    min_lemma_chars: int = 3,
    only_nouns: bool = True,
    keep_duplicates: bool = True,
) -> List[Any]:
    synsets = []

    for keyword in keywords:
        tokens = keyword.split()

        if len(tokens) == 1:
            synset = _token_to_synset(
                keyword,
                min_lemma_chars=min_lemma_chars,
                only_nouns=only_nouns,
            )

            if synset is not None:
                synsets.append(synset)

            continue

        keyword_synsets = _sentence_tokens_to_synsets(
            tokens,
            only_nouns=only_nouns,
            max_lemma_words=max_lemma_words,
            min_lemma_chars=min_lemma_chars,
        )

        if keyword_synsets:
            synsets.extend(keyword_synsets)

    if not keep_duplicates:
        synsets = sorted(list(set(synsets)), key=synsets.index)

    return synsets


def normalize_keywords(
    keywords: List[str],
    max_lemma_words: int = 3,
    min_lemma_chars: int = 3,
    only_nouns: bool = True,
    keep_duplicates: bool = True,
    include_unknown: bool = False,
) -> List[str]:
    normalized_keywords = []

    for keyword in keywords:
        synsets = keywords_to_synsets(
            [keyword],
            only_nouns=False,
            max_lemma_words=max_lemma_words,
            min_lemma_chars=min_lemma_chars,
            keep_duplicates=False,
        )

        if not synsets:
            if include_unknown:
                normalized_keywords.append(keyword)

            continue

        for synset in synsets:
            if only_nouns and synset.pos() != 'n':
                continue

            name = synset.name()
            name = name.split('.')[0]
            name = name.replace('_', ' ')

            normalized_keywords.append(name)

    if not keep_duplicates:
        normalized_keywords = sorted(
            list(set(normalized_keywords)),
            key=normalized_keywords.index,
        )

    return normalized_keywords


def _subsequences_indices(sequence, max_len=None):
    if max_len is None:
        max_len = len(sequence)

    ixs = (
        zip(range(len(sequence)), range(n, len(sequence) + 1))
        for n in range(max_len, 0, -1)
    )

    yield from chain(*ixs)


def _token_to_synset(token, min_lemma_chars=3, only_nouns=True):
    if len(token) < min_lemma_chars:
        return

    if only_nouns:
        word_synonyms = wn.synsets(token, 'n')

    else:
        word_synonyms = wn.synsets(token)

    if word_synonyms:
        return word_synonyms[0]


def _sentence_tokens_to_synsets(
    tokens,
    only_nouns=True,
    max_lemma_words=3,
    min_lemma_chars=3,
):
    synsets = []
    seen = set()

    for start, end in _subsequences_indices(tokens, max_len=max_lemma_words):
        if start in seen or end in seen:
            continue

        wn_candidate = '_'.join(tokens[start: end])
        synset = _token_to_synset(
            wn_candidate,
            min_lemma_chars=min_lemma_chars,
            only_nouns=only_nouns,
        )

        if synset is not None:
            synsets.append(synset)
            seen.update(range(start, end))

    return synsets
