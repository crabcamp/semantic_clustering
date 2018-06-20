from nltk.corpus import wordnet as wn


def _keywords2synsets(keywords, only_nouns=True):
    synsets = []

    for word in keywords:
        if only_nouns:
            word_synonims = wn.synsets(word, 'n')

        else:
            word_synonims = wn.synsets(word)

        if word_synonims:
            synsets.append(word_synonims[0])

    return synsets
