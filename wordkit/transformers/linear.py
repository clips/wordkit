"""Transform orthography."""
import numpy as np

from .base import FeatureTransformer


class LinearTransformer(FeatureTransformer):
    """
    A vectorizer to convert words to vectors based on characters.

    LinearTransformer is meant to be used in models which require a
    measure of orthographic similarity based on visual appearance,
    i.e. the presence or absence of line segments. Such an approach
    is in line with work on word reading, most notably the Interactive
    Activation (IA) models by Mcclelland and Rumelhart.

    The core assumption behind the LinearTransformer is the idea that words
    are read sequentially. Therefore, the LinearTransformer is unable to
    account for transposition or subset effects.

    For example: the words "PAT" and "SPAT" are maximally different according
    to the LinearTransformer, because they don't share any letters in any
    position.

    Parameters
    ----------
    features : dict
        A dictionary of features, where the keys are characters and the
        values are numpy arrays.
    field : str
        The field to retrieve from the incoming dictionaries.
    left : bool, default True
        If this is set to True, all strings will be left-justified. If this
        is set to False, they will be right-justified.

    """

    def __init__(self, features, field, left=True):
        """Convert characters to vectors."""
        if " " not in features:
            features[" "] = np.zeros_like(list(features.values())[0])
        super().__init__(features, field)
        self.vec_len = 0
        self.max_word_length = 0
        self.left = left

    def fit(self, X, y=None):
        """
        Fit the orthographizer by setting the vector length and word length.

        Parameters
        ----------
        X : list of strings or list of dictionaries.
            The input words.

        Returns
        -------
        self : LinearTransformer
            The fitted LinearTransformer instance.

        """
        if type(X[0]) == dict:
            words = [x[self.field] for x in X]
        else:
            words = X
        self._check(words)
        self.max_word_length = max([len(x) for x in words])
        self.vec_len = self.max_word_length * self.dlen
        self._is_fit = True
        return self

    def vectorize(self, x):
        """
        Convert a single word into a vectorized representation.

        Raises a ValueError if the word is too long.

        Parameters
        ----------
        x : dictionary with self.field as key or string.
            The word to vectorize.

        Returns
        -------
        v : numpy array
            A vectorized version of the word.

        """
        if type(x) == dict:
            x = x[self.field]
        v = np.zeros((self.max_word_length, self.dlen))
        if self.left:
            offset = 0
        else:
            offset = self.max_word_length - len(x)
        for idx, c in enumerate(x):
            v[idx+offset] += self.features[c]

        return v.ravel()

    def inverse_transform(self, X):
        """Transform a corpus back to word representations."""
        feature_length = self.vec_len // self.max_word_length
        X_ = X.reshape((X.shape[0], self.max_word_length, feature_length))

        keys, features = zip(*self.features.items())
        keys = [str(x) for x in keys]
        features = np.array(features)

        res = np.linalg.norm(X_[None, :, :] - features[:, None, :], axis=-1)
        res = res.argmin(1)

        return ["".join([keys[idx] for idx in x]) for x in res]
