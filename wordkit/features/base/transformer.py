"""Base classes for transformers."""
from itertools import chain

import numpy as np
import pandas as pd

from .feature_extraction import BaseExtractor


class BaseTransformer(object):
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
        self.field = field

    @property
    def _is_fit(self):
        """
        Whether the transformer has been fit.

        All transformers with feature_names as an attribute have been
        through the fitting stage, which means we can use the presence of
        _is_fit as a shorthand for the transformer being fit.

        Other classes can override this property if they want to redefine
        what it is to be _fit_.

        """
        return hasattr(self, "feature_names")

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
            raise ValueError(
                "The sequence contained illegal features: {0}".format(overlap)
            )

    def _unpack(self, X):
        """Unpack the input data."""
        if isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, pd.DataFrame):
            if self.field is None:
                raise ValueError(
                    "Your field was set to None, but you passed"
                    " a DataFrame. Please pass an explicit field"
                    " when passing a DataFrame."
                )
            return X[self.field].values
        elif isinstance(X[0], dict):
            if self.field is None:
                raise ValueError(
                    "Your field was set to None, but you passed a"
                    " dict. Please pass an explicit field when "
                    "passing a dict."
                )
            return [x[self.field] for x in X]

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

    @property
    def _dtype(self):
        """Dynamically determine the dtype of the features."""
        try:
            v = next(iter(self.features.values()))
            if isinstance(v, np.ndarray):
                return v.dtype
            elif isinstance(v, (tuple, list, float)):
                return np.dtype(type(v[0]))
            else:
                return np.dtype(type(v))
        except AttributeError:
            return np.float

    def transform(self, X, strict=True):
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
        if strict:
            self._validate(X)

        total = np.zeros((len(X), self.vec_len), dtype=self._dtype)
        # Looks silly, but much faster than adding to a list and then
        # turning this into an array.
        for idx, word in enumerate(X):
            total[idx] = self.vectorize(word)

        return total

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FeatureTransformer(BaseTransformer):
    """
    Base class for transformers which have features.

    Parameters
    ----------
    features : dict, tuple of dicts, or FeatureExtractor instance.
        features can either be
            - a dictionary of features, for characters.
            - a tuple of a dictionary of features, for vowels and consonants.
            - an initialized FeatureExtractor instance.

        In the first two cases, the features you input to the Transformer are
        used. In the final case, the FeatureExtractor is used to extract
        features from your input during fitting.

        The choice between pre-defined featues and an is purely a matter of
        convenience. First extracting features using the FeatureExtractor
        leads to the same result as using the FeatureExtractor directly.
    field : str
        The field to retrieve for featurization from incoming records.

    """

    def __init__(self, features, field):
        """Wordkit transformer base class."""
        super().__init__(field)
        if isinstance(features, dict):
            self.features = {k: np.array(v) for k, v in features.items()}
            self.dlen = max([len(x) for x in features.values()])
            self.extractor = None
        elif isinstance(features, tuple):
            self.features = (
                {k: np.array(v) for k, v in features[0].items()},
                {k: np.array(v) for k, v in features[1].items()},
            )
            self.extractor = None
        elif isinstance(features, BaseExtractor) or isinstance(features, type):
            self.features = {}
            if isinstance(features, type):
                features = features()
            self.extractor = features
            self.extractor.field = self.field
        else:
            raise ValueError(
                "The features you passed were not of the "
                "correct type: expected (tuple, dict, "
                "BaseExtractor), got {}".format(type(features))
            )

    def fit(self, X, y=None):
        """Fit the transformer."""
        if self.extractor is not None:
            features = self.extractor.extract(X)
            if not isinstance(features, dict):
                c, v = features
                self.features = (
                    {k: np.array(v) for k, v in c.items()},
                    {k: np.array(v) for k, v in v.items()},
                )
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
