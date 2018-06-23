from nltk.corpus import wordnet as wn

SIMILARITY_METHODS = ['jcn', 'lin', 'path', 'wup']


def get_similarity_method(method):
    method = method.lower()

    if method not in SIMILARITY_METHODS:
        methods_repr = str(SIMILARITY_METHODS)
        raise ValueError('Available values for \'method\' are ' + methods_repr)

    if method == 'jcn':
        return wn.jcn_similarity

    elif method == 'lin':
        return wn.lin_similarity

    elif method == 'path':
        return wn.path_similarity

    elif method == 'wup':
        return wn.wup_similarity


def keywords2synsets(keywords, only_nouns=True):
    synsets = []

    for word in keywords:
        if only_nouns:
            word_synonyms = wn.synsets(word, 'n')

        else:
            word_synonyms = wn.synsets(word)

        if word_synonyms:
            synsets.append(word_synonyms[0])

    return synsets
