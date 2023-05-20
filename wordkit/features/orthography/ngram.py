"""Wickelcoding."""
from itertools import chain

import numpy as np

from wordkit.features.base.transformer import BaseTransformer


class NGramTransformer(BaseTransformer):
    """
    A transformer for Wickelgraphs or Wickelphones.

    Character ngrams are also known under the name of wickelphones or -graphs.

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

    @property
    def _dtype(self):
        return np.int32

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
        self : NGramTransformer
            The transformer itself.

        """
        super().fit(X)
        X = self._unpack(X)
        self.feature_names = set(chain.from_iterable(X))
        grams = set()
        for x in X:
            try:
                _, g = list(zip(*self._decompose(x)))
            except ValueError as e:
                raise ValueError("'{}' did not contain any ngrams.".format(x)) from e
            grams.update(g)

        grams = sorted(grams)
        self.features = {g: idx for idx, g in enumerate(grams)}
        self.inv_features = {v: k for k, v in self.features.items()}
        # The vector length is equal to the number of features.
        self.vec_len = len(self.features)

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
                z[idx] += w
            except KeyError:
                pass

        return z

    def _pad(self, word):
        if self.use_padding:
            padding = ("#",) * (self.n - 1)
            word = tuple(chain(*(padding, word, padding)))
        else:
            word = tuple(word)
        return word

    def _ngrams(self, word):
        """Lazily get all ngrams in a string."""
        word = self._pad(word)
        if len(word) < self.n:
            raise ValueError(
                "You tried to featurize words shorter than "
                "{} characters, please remove these before "
                "featurization, or use padding".format(self.n)
            )

        for i in range(self.n, len(word) + 1):
            yield word[i - self.n : i]

    def _decompose(self, word):
        """Decompose a string into ngrams."""
        grams = tuple(self._ngrams(word))
        return tuple(zip(np.ones(len(grams)), grams))

    def inverse_transform(self, X, threshold=0.9):
        """
        Convert a vector back into its constituent ngrams.

        WARNING: this currently does not work.
        """
        inverted = []
        inverted_features = {v: k for k, v in self.features.items()}

        if not self.use_padding:
            raise ValueError(
                "This function is only supported when use_padding is set to True."
            )

        if np.ndim(X) == 1:
            X = X[None, :]

        for x in X:
            t = []
            for idx in np.flatnonzero(x > threshold):
                t.append(inverted_features[idx])

            cols = list(zip(*t))

            s, e = list(zip(*cols[:-1])), list(zip(*cols[1:]))
            pad = tuple(["#"] * (self.n - 1))
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

    def list_features(self, X):
        """Lists the features for each word."""
        if isinstance(X, np.ndarray):
            for x in X:
                yield tuple([self.inv_features[idx] for idx in np.flatnonzero(x)])
        else:
            X = self._unpack(X)
            for x in X:
                yield tuple(zip(*self._decompose(x)))[1]
