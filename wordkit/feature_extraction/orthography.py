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

    def __init__(self, field=None, include_space=True):
        """Initialize the extractor."""
        super().__init__(field)
        self.include_space = include_space

    def _process(self, symbols):
        """Create one hot encoded features."""
        if self.include_space and " " not in symbols:
            symbols = [" "] + list(symbols)
        features = np.eye(len(symbols), dtype=np.int)
        return dict(zip(symbols, features))
