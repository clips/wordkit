"""Base classes for transformers."""
import numpy as np

from itertools import chain
from sklearn.base import TransformerMixin
from .feature_extraction import BaseExtractor


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

    def _validate(self, X):
        """
        Check whether an input dataset contains illegal features.

        Calculate the difference of the keys of the feature dict and x.
        Raises a ValueError if the result is non-empty.

        Parameters
        ----------
        X : list of strings or list of dicts.
            An input dataset.

        """
        feats = set(chain.from_iterable(X))
        overlap = feats.difference(self.feature_names)
        if overlap:
            raise ValueError("The sequence contained illegal features: {0}"
                             .format(overlap))

    def _unpack(self, X):
        """Unpack the input data."""
        if isinstance(X[0], dict):
            if self.field is None:
                raise ValueError("Your field was set to None, but you passed a"
                                 " dict. Please pass an explicit field when "
                                 "passing a dict.")
            X = [x[self.field] for x in X]

        return X

    def inverse_transform(self, X):
        """Invert the transformation of a transformer."""
        raise NotImplementedError("Base class method.")

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def vectorize(self, x):
        """Vectorize a word."""
        raise NotImplementedError("Base class method.")

    def transform(self, X):
        """
        Transform a list of words.

        Parameters
        ----------
        X : list of string or list of dict
            A list of words.

        Returns
        -------
        features : np.array
            The featurized representations of the input words.

        """
        if not self._is_fit:
            raise ValueError("The transformer has not been fit yet.")
        X = self._unpack(X)
        self._validate(X)

        total = []

        for idx, word in enumerate(X):
            x = self.vectorize(word)
            # This ensures that transformers which return sequences of
            # differing lengths still return non-jagged arrays.
            total.append(x)

        return np.array(total)


class FeatureTransformer(BaseTransformer):
    """
    Base class for transformers which have features.

    Parameters
    ----------
    features : dict, tuple of dicts, or FeatureExtractor instance.
        features can either be
            a dictionary of features, for characters.
            a tuple of a dictionary of features, for vowels and consonants.
            an initialized FeatureExtractor instance.

        In the first two cases, the features you input to the Transformer are
        used. In the final case, the FeatureExtractor is used to extract
        features from your input during fitting.

        The choice between pre-defined featues and an is purely a matter of
        convenience. First extracting features using the FeatureExtractor
        leads to the same result as using the FeatureExtractor directly.

    field : str
        The field to retrieve for featurization from incoming records.

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
        elif isinstance(features, BaseExtractor) or isinstance(features, type):
            self.features = {}
            if isinstance(features, type):
                features = features()
            self.extractor = features
            self.extractor.field = self.field

    def fit(self, X, y=None):
        """Fit the transformer."""
        if self.extractor:
            features = self.extractor.extract(X)
            if not isinstance(features, dict):
                c, v = features
                self.features = ({k: np.array(v) for k, v in c.items()},
                                 {k: np.array(v) for k, v in v.items()})
            else:
                self.dlen = max([len(x) for x in features.values()])
                self.features = {k: np.array(v) for k, v in features.items()}

        if isinstance(self.features, tuple):
            for x in self.features:
                if " " not in self.features:
                    v = next(iter(x.values()))
                    x[" "] = np.zeros_like(v)
        else:
            if " " not in self.features:
                v = next(iter(self.features.values()))
                self.features[" "] = np.zeros_like(v)

        return self
