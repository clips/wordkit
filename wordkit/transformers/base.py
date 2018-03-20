"""Base classes for transformers."""
import numpy as np

from itertools import chain
from sklearn.base import TransformerMixin
from ..feature_extraction.base import BaseExtractor
from ..feature_extraction.phonology import BasePhonemeExtractor


class BaseTransformer(TransformerMixin):
    """
    Base class for transformers without input features.

    Parameters
    ----------
    field : string
        The field to extract from the incoming dictionaries. For example, if
        you want to featurize orthographic forms, you can pass "orthography"
        to field. Most featurizers accept both "orthography" and "phonology"
        as possible fields.

    """

    def __init__(self, field):
        """Initialize the transformer."""
        self._is_fit = False
        self.features = None
        self.field = field

    def _check(self, x):
        """
        Check whether a feature string contains illegal features.

        Calculate the difference of the keys of the feature dict and x.
        Raises a ValueError if the result is non-empty.

        Parameters
        ----------
        x : string
            An input string.

        """
        x = set(chain.from_iterable(x))
        overlap = x.difference(set(self.features.keys()))
        if overlap:
            raise ValueError("The sequence contained illegal features: {0}"
                             .format(overlap))

    def inverse_transform(self, X):
        """Invert the transformation of a transformer."""
        raise NotImplementedError("Base class method.")

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def vectorize(self, x):
        """Vectorize a word."""
        raise NotImplementedError("Base class method.")

    def transform(self, words):
        """
        Transform a list of words.

        Parameters
        ----------
        words : list of string or list of dict
            A list of words.

        Returns
        -------
        features : np.array
            The featurized representations of the input words.

        """
        if not self._is_fit:
            raise ValueError("The transformer has not been fit yet.")
        total = np.zeros((len(words), self.vec_len))

        for idx, word in enumerate(words):
            x = self.vectorize(word)
            # This ensures that transformers which return sequences of
            # differing lengths still return non-jagged arrays.
            total[idx, :len(x)] = x

        return np.array(total)


class FeatureTransformer(BaseTransformer):
    """
    Base class for transformers which have features.

    Parameters
    ----------
    features : dict or tuple of dicts
        A key to array mapping, or a collection of key to array mappings.
    field : str
        The field to retrieve for featurization from incoming records.
    vec_len : int, default 0
        The vector length.

    """

    def __init__(self,
                 features,
                 field):
        """Wordkit transformer base class."""
        super().__init__(field)
        if isinstance(features, dict):
            self.features = {k: np.array(v) for k, v in features.items()}
            self.dlen = max([len(x) for x in features.values()])
            self.extractor = None
        elif isinstance(features, tuple):
            self.features = ({k: np.array(v) for k, v in features[0].items()},
                             {k: np.array(v) for k, v in features[1].items()})
            self.extractor = None
        elif isinstance(features, BaseExtractor):
            self.features = {}
            self.extractor = features
            self.extractor.field = self.field

    def fit(self, X, y=None):

        if self.extractor:
            features = self.extractor.extract(X)
            if not isinstance(self.extractor, BasePhonemeExtractor):
                self.dlen = max([len(x) for x in features.values()])
                self.features = {k: np.array(v) for k, v in features.items()}
            else:
                c, v = features
                self.features = ({k: np.array(v) for k, v in c.items()},
                                 {k: np.array(v) for k, v in v.items()})

        return self._fit(X)
