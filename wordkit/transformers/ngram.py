"""Transform orthography."""
from .wickel import WickelTransformer
from itertools import combinations


class OpenNGramTransformer(WickelTransformer):
    """
    Vectorizes words as the n-combination of all letters in the word.

    The OpenNGramTransformer tries to take into account transposition effects
    by using the unordered n-combination of the characters in the word.

    For example, using n==2, the word "SALT" will be represented as
    {"SA", "SL", "ST", "AL", "AT", "LT"}, while "SLAT" will be represented as
    {"SA", "SL", "ST", "LA", "AT", "LT"}. Note that these two representations
    only differ in a single bigram, and therefore are highly similar
    according to this encoding scheme.

    Note that this transformer is not limited to orthographic strings.
    It can also handle phonological strings, by setting the fields
    appropriately.

    Parameters
    ----------
    n : int
        The value of n to use for the n-combations.
    field : string
        The field to which to apply this transformer.

    """

    def __init__(self, n, field):
        """Initialize the transformer."""
        super().__init__(n, field)

    @staticmethod
    def _ngrams(word, n):
        """Get all unordered n-combinations of characters in a word."""
        return set(combinations(word, n))
