"""Transform orthography."""
from .wickel import WickelTransformer
from itertools import combinations
from functools import reduce


class OpenNGramTransformer(WickelTransformer):
    r"""
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

    If you use the OpenNGramTransformer, please cite:

        @article{grainger2004modeling,
          title={Modeling letter position coding in printed word perception.},
          author={Grainger, Jonathan and Van Heuven, Walter JB},
          year={2004},
          publisher={Nova Science Publishers}
        }

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

    def _decompose(self, word):
        """Get all unordered n-combinations of characters in a word."""
        return set(combinations(word, self.n))


class ConstrainedOpenNGramTransformer(WickelTransformer):
    r"""
    Vectorizes words as the n-combination of letters within some window.

    The ConstrainedOpenNGramTransformer is extremely similar to the
    OpenNGramTransformer, above, but only calculates the overlap between
    letters within some pre-defined window.

    The ConstrainedOpenNGramTransformer is equivalent to the
    OpenNGramTransformer for short words, but will produce different results
    for longer words.

    If you use the ConstrainedOpenNGramTransformer, please cite:

    @article{whitney2001brain,
      title={How the brain encodes the order of letters in a printed word:
             The SERIOL model and selective literature review},
      author={Whitney, Carol},
      journal={Psychonomic Bulletin \& Review},
      volume={8},
      number={2},
      pages={221--243},
      year={2001},
      publisher={Springer}
    }

    Parameters
    ----------
    n : int
        The value of n to use for the n-combations.
    window : int
        The maximum distance between two letters.
    field : string
        The field to which to apply this transformer.
    use_padding : bool
        Whether to pad the words with a single "#" character.

    """

    def __init__(self, n, window, field, use_padding=False):
        """Initialize the transformer."""
        super().__init__(n, field)
        self.window = window
        self.use_padding = use_padding

    def _decompose(self, word):
        """Get all unordered n-combinations of characters in a word."""
        grams = self._ngrams(word, self.window+1, 1 if self.use_padding else 0)
        combs = (combinations(x, self.n) for x in grams)
        return reduce(set.union, combs, set())
