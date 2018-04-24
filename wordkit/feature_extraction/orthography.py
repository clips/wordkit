"""Functions for extracting orthographical features."""
import numpy as np

from .base import BaseExtractor


class OneHotCharacterExtractor(BaseExtractor):
    """
    Convert your characters to one hot features.

    Note: if your data contains diacritics, such as 'é', you might be better
    off normalizing these. The OneHotCharacterExtractor will assign these
    completely dissimilar representations.

    Example
    -------
    >>> from wordkit.feature_extraction import OneHotCharacterExtractor
    >>> words = ["c'est", "stéphan"]
    >>> o = OneHotCharacterExtractor(field=None)
    >>> o.extract(words)

    """

    def _process(self, symbols):
        """Create one hot encoded features."""
        features = np.eye(len(symbols), dtype=np.int)
        return dict(zip(symbols, features))
