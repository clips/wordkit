"""Wickelcoding."""
import numpy as np

from .base import BaseTransformer


class WickelTransformer(BaseTransformer):
    """
    A transformer for naive Wickelfeatures.

    Wickelfeatures are more commonly known under the name of character ngrams.
    Wickelfeatures assume that words are represented as unordered ngrams of
    characters. For each character in the word, the n letters on the left
    and right are added to it. This introduces a modicum of context into the
    letter representations.

    For example, using a n=1 the word "CAT" gets represented as "#CA", "CAT",
    and "AT#". Note the addition of the beginning and end markers.

    Parameters
    ----------
    n : int
        The value of n to use in the character ngrams.
    field : str
        The field to which to apply this featurizer.

    """

    def __init__(self, n, field):
        """Initialize the transformer."""
        super().__init__(field)
        self.n = n
        self.vec_len = 0

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
        if type(X[0]) == dict:
            words = [x[self.field] for x in X]
        else:
            words = X
        grams = set()
        for x in words:
            grams.update(self._ngrams(x, self.n))
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
        if type(x) == dict:
            x = x[self.field]
        z = np.zeros(self.vec_len)
        indices = [self.features[g] for g in self._ngrams(x, self.n)]
        z[indices] = 1

        return z

    @staticmethod
    def _ngrams(word, n):
        """Lazily get all ngrams in a string."""
        prepend = ("#",) * n
        col = prepend + tuple(word) + prepend
        for i in range(n, len(col)-n):
            yield col[i-n: i+(n+1)]
