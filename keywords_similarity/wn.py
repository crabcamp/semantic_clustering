from functools import lru_cache

from nltk.corpus import wordnet as wn

SIMILARITY_METHODS = ('path', 'path_cached', 'wup', 'wup_cached')


def get_similarity_function(method):
    if method not in SIMILARITY_METHODS:
        methods_repr = str(SIMILARITY_METHODS)
        raise ValueError('Available values for \'method\' are ' + methods_repr)

    if method == 'path':
        return wn.path_similarity

    elif method == 'path_cached':
        return path_similarity_cached

    elif method == 'wup':
        return wn.wup_similarity

    elif method == 'wup_cached':
        return wup_similarity_cached


def keywords_to_synsets(keywords, only_nouns=True, keep_duplicates=True):
    synsets = []

    for word in keywords:
        if only_nouns:
            word_synonyms = wn.synsets(word, 'n')

        else:
            word_synonyms = wn.synsets(word)

        if word_synonyms:
            synsets.append(word_synonyms[0])

    if not keep_duplicates:
        synsets = sorted(list(set(synsets)), key=synsets.index)

    return synsets


@lru_cache(maxsize=1048576)
def path_similarity_cached(synset_1, synset_2):
    synsets = sorted([synset_1, synset_2])

    return wn.path_similarity(*synsets)


@lru_cache(maxsize=1048576)
def wup_similarity_cached(synset_1, synset_2):
    synsets = sorted([synset_1, synset_2])

    return wn.wup_similarity(*synsets)
