"""Holographic feature methods."""
import numpy as np

from ..base import BaseTransformer
from ..orthography import NGramTransformer
from itertools import chain, combinations


class HolographicTransformer(BaseTransformer):
    """
    Holographic features for representing words.

    """

    def __init__(self, vec_len, field=None):
        """Initialize the transformer."""
        super().__init__(field)
        self.vec_len = vec_len

    def fit(self, X):
        """Input to HolographicTransformer is a tree."""
        super().fit(X)
        X = self._unpack(X)
        X = [self.hierarchify(x) for x in X]
        # Flatten lists.
        features = set(chain(*chain(*X)))
        vectors = self.generate((len(features), self.vec_len))
        self.features = dict(zip(features, vectors))
        self.feature_names = list(self.features)
        n_positions = max([max(len(g) for g in x) for x in X])
        position_vectors = self.generate_positions((n_positions, self.vec_len))
        self.positions = dict(zip(range(n_positions), position_vectors))
        self._is_fit = True

    def vectorize(self, x):
        """Vectorize a hierarchical item."""
        x = self.hierarchify(x)
        vec = np.zeros(self.vec_len)
        for item in x:
            z = None
            for idx, char in enumerate(item):
                if z is None:
                    z = self.compose(char, idx)
                else:
                    z = self.add(z, self.compose(char, idx))
            vec = self.add(vec, z)

        return vec


class NGramMixIn(object):

    def hierarchify(self, x):
        padding = self.use_padding * (self.n - 1)
        return list(map(tuple, NGramTransformer._ngrams(x, self.n, padding)))


class OpenNGramMixIn(object):

    def hierarchify(self, x):
        return list(map(tuple, combinations(x, self.n)))


class LinearMixIn(object):

    def hierarchify(self, x):
        return [tuple(x)]


class ConstrainedOpenNGramMixIn(object):

    def hierarchify(self, x):
        t = []
        if self.use_padding:
            x = "#{}#".format(x)
        for idx in range(len(x)):
            subword = x[idx:idx+(self.window+2)]
            focus_letter = x[idx]
            for c in combinations(subword, self.n):
                if c[0] != (focus_letter):
                    continue
                t.append(tuple(c))
        return t
