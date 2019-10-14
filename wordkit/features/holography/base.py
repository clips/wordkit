"""Holographic feature methods."""
from ..base import BaseTransformer
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
        X = [list(self.hierarchify(x)) for x in X]
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
        vec = []
        for item in x:
            for idx, char in enumerate(item):
                vec.append(self.compose(char, idx))
        return self.add(vec)


class LinearMixin(object):

    def hierarchify(self, x):
        return [tuple(x)]


class NGramMixin(object):

    def _pad(self, word):
        if self.use_padding:
            padding = ("#",) * (self.n - 1)
            word = tuple(chain(*(padding, word, padding)))
        else:
            word = tuple(word)
        return word

    def hierarchify(self, word):
        """Turn a word into an ngram hierarchy"""
        word = self._pad(word)
        if len(word) < self.n:
            raise ValueError("You tried to featurize words shorter than "
                             "{} characters, please remove these before "
                             "featurization, or use padding".format(self.n))

        for i in range(self.n, len(word)+1):
            yield word[i-self.n: i]


class OpenNgramMixin(NGramMixin):

    def hierarchify(self, word):
        """Turn a word into an ngram hierarchy"""
        return combinations(word, self.n)


class ConstrainedOpenNgramMixin(NGramMixin):

    def hierarchify(self, word):
        """Turn a word into an ngram hierarchy"""
        word = self._pad(word)
        for idx in range(len(word)):
            subword = word[idx:idx+(self.window+1)]
            for x in combinations(subword[1:], self.n-1):
                yield tuple(chain(*(subword[0], chain(*x))))
