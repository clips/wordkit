import numpy as np
import jellyfish

from wordkit.readers import Celex
from wordkit.transformers import ConstrainedOpenNGramTransformer, OpenNGramTransformer, WickelTransformer, LinearTransformer
from wordkit.features import fourteen
from string import ascii_lowercase
import regex as re
from collections import defaultdict
from itertools import combinations
from pyxdameraulevenshtein import damerau_levenshtein_distance


def testo_2(word, lexicon):

    z = {}

    for x in lexicon:
        if x == word:
            continue
        z[x] = jellyfish.levenshtein_distance(word, x)

    return list(sorted(z.items(), key=lambda x: x[1]))


def testo(words):

    old_words = defaultdict(list)

    # Damerau-Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    for a, b in combinations(words, 2):
        dist = jellyfish.levenshtein_distance(a, b)
        old_words[a].append(dist)
        old_words[b].append(dist)

    return dict(old_words)


if __name__ == "__main__":

    X = Celex("/Users/stephantulkens/Documents/corpora/celex/epl.cd",
              filter_function=lambda x: not set(x['orthography']) - set(ascii_lowercase),
              fields=('orthography',)).transform()

    b_t = ConstrainedOpenNGramTransformer(2, 2, field='orthography')
    o_t = OpenNGramTransformer(2, field='orthography')
    w = WickelTransformer(3, field='orthography')

    l = LinearTransformer(fourteen, left=False, field='orthography')
    p = l.fit_transform(X)
