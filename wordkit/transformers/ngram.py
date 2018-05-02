"""Transform orthography."""
import numpy as np
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

    def __init__(self, n, field=None):
        """Initialize the transformer."""
        super().__init__(n, field)

    def _decompose(self, word):
        """Get all unordered n-combinations of characters in a word."""
        combs = set(combinations(word, self.n))
        if not combs:
            combs = [word]
        return zip(np.ones(len(combs)), combs)


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

    def __init__(self, n, window, field=None, use_padding=False):
        """Initialize the transformer."""
        super().__init__(n, field)
        self.window = window
        self.use_padding = use_padding

    def _decompose(self, word):
        """Get all unordered n-combinations of characters in a word."""
        grams = self._ngrams(word,
                             self.window+1,
                             1 if self.use_padding else 0,
                             strict=False)
        combs = (combinations(x, self.n) for x in grams)
        result = list(reduce(set.union, combs, set()))
        return zip(np.ones(len(result)), result)


class WeightedOpenBigramTransformer(ConstrainedOpenNGramTransformer):
    r"""
    A transformer for weighted open bigrams.

    A weighted open bigram is an open bigram with a distance-dependent weight.
    The weight assigned to each bigram depends on the distance between the
    constituent letters of said bigram.

    That is: if the letters of a bigram are contiguous, their weight is higher
    than the weight of two letters that happen to be further away from each
    other.

    The WeightedOpenBigramTransformer can only handle bigrams, because there
    is no nice way to assign values to trigrams based on their contiguity.

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
    field : string
        The field to apply this transformer to.

    weights : tuple
        The weights to apply at each distance. The first weight is applied at
        distance one, the second at distance two etc. Any letters which are
        have a distance greater than (len(weights) + 1) are given a weight of 0

    use_padding : bool, default False
        Whether to use padding.

    """

    def __init__(self, weights, field=None, use_padding=False):
        """Init the object."""
        super().__init__(2, len(weights), field, use_padding)
        self.weights = weights

    def _decompose(self, word):
        """Decompose a word into its consituent letters."""
        grams = self._ngrams(word,
                             self.window+1,
                             1 if self.use_padding else 0,
                             strict=False)
        gram_index = list(range(self.window+1))

        for gram in grams:
            combs = combinations(zip(gram_index, gram), self.n)
            for (a, l1), (b, l2) in combs:
                yield self.weights[abs(a-b)-1], (l1, l2)
