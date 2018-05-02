"""Wickelcoding."""
import numpy as np

from .base import BaseTransformer


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
        grams = sorted(grams)
        self.features = {g: idx for idx, g in enumerate(grams)}
        # The vector length is equal to the number of features.
        self.vec_len = len(self.features)
        self.feature_names = set(self.features.keys())
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
        weights, indices = zip(*[(w, self.features[g])
                               for w, g in self._decompose(x)])

        z[list(indices)] = weights

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
