from nltk.corpus import wordnet as wn

SIMILARITY_METHODS = ('path', 'wup')


def get_similarity_function(method):
    method = method.lower()

    if method not in SIMILARITY_METHODS:
        methods_repr = str(SIMILARITY_METHODS)
        raise ValueError('Available values for \'method\' are ' + methods_repr)

    if method == 'path':
        return wn.path_similarity

    elif method == 'wup':
        return wn.wup_similarity


def keywords2synsets(keywords, only_nouns=True, keep_duplicates=True):
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
