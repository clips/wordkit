"""Wickelcoding."""
import numpy as np

from .base import BaseTransformer
from math import ceil
from itertools import product


class WickelTransformer(BaseTransformer):
    """
    A transformer for Wickelgraphs or Wickelphones.

    Wickelgraphs are more commonly known under the name of character ngrams.

    Wickelgraphs assume that words are represented as unordered ngrams of
    characters. This introduces some idea of context into the letter
    representations, and allows the model to generalize beyond specific letter
    positions.

    For example, using n=3 the word "CAT" gets represented as "#CA", "CAT",
    and "AT#". Note the addition of the beginning and end markers. If you
    set use_padding to False, these end markers get removed, and "CAT" becomes
    a single ngram ("CAT").

    Note that trying to featurize words which are smaller than the ngram size
    will throw a ValueError. This can only happen if padding is set to False.

    Parameters
    ----------
    n : int
        The value of n to use in the character ngrams.

    field : str
        The field to which to apply this featurizer.

    use_padding : bool, default True
        Whether to include "#" characters as padding characters.

    """

    def __init__(self, n, field=None, use_padding=True):
        """Initialize the transformer."""
        super().__init__(field)
        self.n = n
        self.use_padding = use_padding

    def fit(self, X):
        """
        Fit the transformer by finding all grams in the input data.

        Parameters
        ----------
        X : dictionary of with 'orthography' as key or list of strings.
            This is usually the output of a wordkit reader, but can also
            simply be a list of strings.

        Returns
        -------
        self : WickelTransformer
            The transformer itself.

        """
        if type(X[0]) == dict:
            words = [x[self.field] for x in X]
        else:
            words = X
        grams = set()
        for x in words:
            g = list(zip(*self._decompose(x)))
            if not g:
                raise ValueError("{} did not contain any ngrams."
                                 "".format(x))
            grams.update(g[1])

        grams = sorted(grams)
        self.features = {g: idx for idx, g in enumerate(grams)}
        # The vector length is equal to the number of features.
        self.vec_len = len(self.features)
        self.feature_names = set(self.features.keys())

        return self

    def vectorize(self, x):
        """
        Convert a single word into a vectorized representation.

        Raises a ValueError if the word is too long.

        Parameters
        ----------
        x : string or dictionary
            The word to convert.

        Returns
        -------
        v : numpy array
            A vectorized representation of the input word.

        """
        z = np.zeros(self.vec_len)
        for w, g in self._decompose(x):

            idx = self.features[g]
            z[idx] = max(z[idx], w)

        return z

    @staticmethod
    def _ngrams(word, n, num_padding, strict=True):
        """Lazily get all ngrams in a string."""
        if num_padding:
            padding = ("#",) * num_padding
            word = padding + tuple(word) + padding
        if len(word) < n:
            if strict:
                raise ValueError("You tried to featurize words shorter than "
                                 "{} characters, please remove these before "
                                 "featurization, or use padding".format(n))
            else:
                yield word

        for i in range(n, len(word)+1):
            yield word[i-n: i]

    def _decompose(self, word):
        """Decompose a string into ngrams."""
        grams = self._ngrams(word,
                             self.n,
                             self.n - 1 if self.use_padding else 0)
        grams = list(grams)
        return list(zip(np.ones(len(grams)), grams))


class WickelFeatureTransformer(WickelTransformer):
    """A transformer for WickelFeatures."""

    def __init__(self,
                 n,
                 num_units,
                 field=None,
                 use_padding=True,
                 proportion=.38):
        """Initialize the transformer."""
        super().__init__(n, field, use_padding)
        self.num_units = num_units
        self.proportion = proportion

    def fit(self, X):
        """
        Fit the orthographizer by setting the vector length and word length.

        Parameters
        ----------
        X : dictionary of with 'orthography' as key or list of strings.
            This is usually the output of a wordkit reader, but can also
            simply be a list of strings.

        Returns
        -------
        self : WickelTransformer
            The transformer itself.

        """
        super().fit(X)

        # Assign each unit
        feature_matrix = np.zeros((len(self.features), self.num_units))
        feature_values = [list(set(x)) for x in zip(*self.features)]
        num_values = [ceil(len(x) * self.proportion) for x in feature_values]
        for col in range(self.num_units):
            triples = []
            for num, x in zip(num_values, feature_values):
                triples.append(np.random.choice(x, size=num, replace=False))
            all_triples = list(product(*triples))
            for triple in all_triples:
                try:
                    feature_matrix[self.features[triple], col] = 1
                except KeyError:
                    pass

        self.features = {k: feature_matrix[v]
                         for k, v in self.features.items()}
        self.vec_len = self.num_units
        self._is_fit = True
        return self

    def vectorize(self, x):
        """
        Convert a single word into a vectorized representation.

        Raises a ValueError if the word is too long.

        Parameters
        ----------
        x : string or dictionary
            The word to convert.

        Returns
        -------
        v : numpy array
            A vectorized representation of the input word.

        """
        z = np.max([self.features[g] for w, g in self._decompose(x)], axis=0)
        return z
