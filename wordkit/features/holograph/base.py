"""Holographic feature methods."""
import numpy as np

from ..base import BaseTransformer
from ..orthography import WickelTransformer
from itertools import chain


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
            z = np.zeros(self.vec_len)
            for idx, char in enumerate(item):
                z = self.add(z, self.compose(char, idx))
            vec = self.add(vec, z)

        return vec


class NGramMixIn(object):

    def hierarchify(self, x):
        return [tuple(x) for x in WickelTransformer._ngrams(x, self.n, 0)]


class OpenNGramMixIn(object):

    def hierarchify(self, X):
        raise NotImplementedError()


class LinearMixIn(object):

    def hierarchify(self, x):
        return [tuple(x)]
