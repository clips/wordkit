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
    use_padding : bool, default True
        Whether to include "#" characters as padding characters.

    """

    def __init__(self, n, field, use_padding=True):
        """Initialize the transformer."""
        super().__init__(field)
        self.n = n
        self.vec_len = 0
        self.use_padding = use_padding

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
            grams.update(list(zip(*self._decompose(x)))[1])
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
        weights, indices = zip(*[(w, self.features[g])
                               for w, g in self._decompose(x)])

        z[list(indices)] = weights

        return z

    @staticmethod
    def _ngrams(word, n, num_padding):
        """Lazily get all ngrams in a string."""
        if num_padding:
            padding = ("#",) * num_padding
            word = padding + tuple(word) + padding

        for i in range(n, len(word)+1):
            yield word[i-n: i]

    def _decompose(self, word):
        """Decompose a string into ngrams."""
        grams = self._ngrams(word, self.n, self.n-1 if self.use_padding else 0)
        grams = list(grams)
        return list(zip(np.ones(len(grams)), grams))
