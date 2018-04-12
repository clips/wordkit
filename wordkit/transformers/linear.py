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
    features : dict, or FeatureExtractor instance.
        features can either be
            a dictionary of features, for characters.
            an initialized FeatureExtractor instance.

        In the first case, the features you input to the Transformer are
        used. In the final case, the FeatureExtractor is used to extract
        features from your input during fitting.

        The choice between pre-defined featues and an is purely a matter of
        convenience. First extracting features using the FeatureExtractor
        leads to the same result as using the FeatureExtractor directly.

    field : str
        The field to retrieve from the incoming dictionaries.

    left : bool, default True
        If this is set to True, all strings will be left-justified. If this
        is set to False, they will be right-justified.

    """

    def __init__(self, features, field, left=True):
        """Convert characters to vectors."""
        super().__init__(features, field)
        self.max_word_length = 0
        self.left = left

    def _fit(self, X):
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
        if " " not in self.features:
            self.features[" "] = np.zeros_like(list(self.features.values())[0])
        if type(X[0]) == dict:
            words = [x[self.field] for x in X]
        else:
            words = X
        self.feature_names = set(self.features.keys())
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
        if len(x) > self.max_word_length:
            raise ValueError("Your word is too long")
        v = np.zeros((self.max_word_length, self.dlen))
        if self.left:
            x = x.ljust(self.max_word_length)
        else:
            x = x.rjust(self.max_word_length)
        for idx, c in enumerate(x):
            v[idx] += self.features[c]

        return v.ravel()

    def inverse_transform(self, X):
        """Transform a corpus back to word representations."""
        feature_length = self.vec_len // self.max_word_length
        X_ = X.reshape((-1, self.max_word_length, feature_length))

        keys, features = zip(*self.features.items())
        keys = [str(x) for x in keys]
        features = np.array(features)

        for x in X_:
            res = np.linalg.norm(x[:, None, :] - features[None, :, :], axis=-1)
            res = res.argmin(1)

            yield "".join([keys[idx] for idx in res]).strip()
