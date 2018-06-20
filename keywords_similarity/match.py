import numpy as np
from nltk.corpus import wordnet as wn


SIMILARITY_MEASURE = {
    'jcn': wn.jcn_similarity,
    'lin': wn.lin_similarity,
    'path': wn.path_similarity,
    'wup': wn.wup_similarity,
}


def _calculate_similarity_matrix(synsets_1, synsets_2, method='wup'):
    if method.lower() not in SIMILARITY_MEASURE:
        sim_funcs = ', '.join('\'' + key + '\'' for key in SIMILARITY_MEASURE)
        err_msg = 'Available values for \'method\' are [{}]'.format(sim_funcs)

        raise ValueError(err_msg)

    similarity_function = SIMILARITY_MEASURE[method]

    n_1, n_2 = len(synsets_1), len(synsets_2)

    similarity_matrix = np.zeros((n_1, n_2))

    for i in range(n_1):
        for j in range(n_2):
            s_i, s_j = synsets_1[i], synsets_2[j]
            similarity_matrix[i, j] = similarity_function(s_i, s_j)

    return similarity_matrix


def _keywords2synsets(keywords, only_nouns=True):
    synsets = []

    for word in keywords:
        if only_nouns:
            word_synonyms = wn.synsets(word, 'n')

        else:
            word_synonyms = wn.synsets(word)

        if word_synonyms:
            synsets.append(word_synonyms[0])

    return synsets



