"""Wickelcoding."""
import numpy as np

from ..base.transformer import BaseTransformer
from math import ceil
from itertools import product, chain


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
        super().fit(X)
        X = self._unpack(X)
        self.feature_names = set(chain.from_iterable(X))
        grams = set()
        for x in X:
            g = list(zip(*self._decompose(x)))
            if not g:
                raise ValueError("{} did not contain any ngrams."
                                 "".format(x))
            grams.update(g[1])

        grams = sorted(grams)
        self.features = {g: idx for idx, g in enumerate(grams)}
        # The vector length is equal to the number of features.
        self.vec_len = len(self.features)
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
        z = np.zeros(self.vec_len)
        for w, g in self._decompose(x):
            try:
                idx = self.features[g]
            except KeyError:
                raise ValueError("You passed a word containing an ngram which"
                                 " was not in the training data: {}"
                                 "".format(g))
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

    def inverse_transform(self, X, threshold=.9):
        """Convert a vector back into its constituent ngrams."""
        inverted = []
        inverted_features = {v: k for k, v in
                             self.features.items()}

        if not self.use_padding:
            raise ValueError("This function is only supported when use_padding"
                             " is set to True.")

        for x in X:
            t = []
            for idx in np.flatnonzero(x > threshold):
                t.append(inverted_features[idx])

            cols = list(zip(*t))

            s, e = list(zip(*cols[:-1])), list(zip(*cols[1:]))
            pad = tuple(["#"] * (self.n-1))
            f = list(set(s) - (set(e) - {pad}))[0]
            word = []
            while len(word) < len(t):
                idx = s.index(f)
                word.append(t[idx][0])
                f = e[idx]
            else:
                word.extend(t[idx][1:])

            inverted.append("".join(word).strip("#"))

        return inverted


class WickelFeatureTransformer(WickelTransformer):
    """
    A transformer for WickelFeatures.

    This transformer behaves more or less the same as the WickelTransformer,
    above, but has 2 advantages: first, it assigns a higher similarity to
    ngrams which have more overlap. Second, it usually leads to spaces with
    smaller dimensionalities.

    Parameters
    ----------
    n : int
        The value of n to use in the character ngrams.

    num_units : int
        The number of units with which to represent each individual character
        ngram. This number is also equal to the output dimensionality.

    field : str or None
        The field on which this transformer operates.

    use_padding : bool
        Whether to use padded or non-padded ngrams.

    proportion : float
        This number approximately encodes the number of units each character
        activates. For example, given a proportion of .38, about 10 of 26
        letters in the alphabet will be assigned to a single unit.

    """

    def __init__(self,
                 n,
                 num_units,
                 field=None,
                 use_padding=True,
                 proportion=.38):
        """Initialize the transformer."""
        super().__init__(n, field, use_padding)
        self.num_units = num_units
        assert .0 < proportion < 1.0
        self.proportion = proportion

    def fit(self, X):
        """
        Fit the featurizer by setting the vector length and word length.

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
        self._is_fit = False
        # Assign each feature an number of units based on the individual
        # characters this feature consists of.
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

    def inverse_transform(self, X):
        """Not implemented."""
        raise NotImplemented("Not implemented because probably impossible.")
