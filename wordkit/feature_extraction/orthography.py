"""Functions for extracting orthographical features."""
import numpy as np

from .base import BaseExtractor


class OneHotCharacterExtractor(BaseExtractor):
    """Convert your characters to one hot features."""

    def _process(self, symbols):
        """Create one hot encoded features."""
        features = np.eye(len(symbols), dtype=np.int)
        return dict(zip(symbols, features))
