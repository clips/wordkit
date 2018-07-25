"""Wordnet semantics."""
import numpy as np
from ..base import BaseTransformer

from nltk.corpus import wordnet as wn
from itertools import chain


class OneHotSemantics(BaseTransformer):
    """
    Code each semantic node as a separate symbol.

    The simplest semantic representation, assumes no overlap between semantics.

    Parameters
    ----------
    field : str or None
        The field to use.
    """

    def __init__(self, field=None):
        """Initialize the transformer."""
        super().__init__(field)

    def fit(self, X):
        """Fit the transformer to semantics."""
        super().fit(X)
        X = self._unpack(X)
        self.feature_names = set()
        for x in X:
            self.feature_names.update(x)

        self.features = {k: idx for idx, k in enumerate(self.feature_names)}
        self.vec_len = len(self.feature_names)
        self._is_fit = True
        return self

    def vectorize(self, x):
        """Vectorize some data."""
        s = np.zeros(self.vec_len)
        s[[self.features[s] for s in x]] = 1
        return s


class HypernymSemantics(BaseTransformer):
    """
    Create semantic vectors by one-hot encoding meronym relations.

    This is an implementation of the semantics approach from the
    Triangle model.

    If you use it, please cite:
    @techreport{harm2002building,
      title={Building large scale distributed semantic
             feature sets with WordNet},
      author={Harm, MW},
      year={2002},
      institution={Technical Report PDP-CNS-02-1, Carnegie Mellon University}
    }



    """

    def __init__(self, field=None, use_meronyms=True):
        """Initialize the transformer."""
        super().__init__(field)
        self.use_meronyms = use_meronyms

    def fit(self, X):
        """Fit the transformer."""
        super().fit(X)
        X = self._unpack(X)

        meronyms = set()
        for x in X:
            for (offset, pos) in x:
                s = wn.synset_from_pos_and_offset(pos, int(offset))
                meronyms.update(self.recursive_related(s))

        self.feature_names = set(chain.from_iterable(X))
        self.features = {k: idx for idx, k in enumerate(meronyms)}
        self.vec_len = len(self.features)
        self._is_fit = True
        return self

    def vectorize(self, x):
        """Vectorize a word."""
        vec = np.zeros(self.vec_len)
        for (offset, pos) in x:
            s = wn.synset_from_pos_and_offset(pos, int(offset))
            s = [self.features[s] for s in self.recursive_related(s)]
            vec[s] = 1

        return vec

    def recursive_related(self, synset):
        """Get recursive meronyms."""
        def find_related(synset, use_meronyms):
            """Recursive method."""
            if use_meronyms:
                related = synset.part_meronyms()
            else:
                related = []
            for x in synset.hypernyms():
                related.append(x)
                related.extend(find_related(x, use_meronyms=use_meronyms))

            return related

        result = find_related(synset, self.use_meronyms)
        return {(s.offset(), s.pos()) for s in result}
