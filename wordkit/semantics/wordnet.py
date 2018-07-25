"""Wordnet semantics."""
import numpy as np
from ..base import BaseTransformer

from nltk.corpus import wordnet as wn
from itertools import chain
from collections import Counter


class OneHotSemanticsTransformer(BaseTransformer):
    """
    Code each semantic node as a separate symbol.

    The simplest semantic representation, assumes no overlap between different
    semantic nodes.
    Nevertheless, lemmas can get quite sensible representations because of
    varying amounts of overlap between their linked synsets.

    Parameters
    ----------
    field : str or None
        The field to use.

    prune : bool
        If this is set to True, semantic nodes which occur only once will be
        removed. This will also skip the validation step, and may lead to
        errors if any synsets are added.

    """

    def __init__(self, field=None, prune=True):
        """Initialize the transformer."""
        super().__init__(field)
        self.prune = prune

    def _validate(self, X):
        """
        Validate the input.

        This is overriden because this transformer can remove things which
        only occur once.
        """
        if not self.prune:
            super()._validate(X)

    def fit(self, X):
        """Fit the transformer to semantics."""
        super().fit(X)
        X = self._unpack(X)
        self.feature_names = Counter()
        for x in X:
            self.feature_names.update(x)

        if self.prune:
            self.feature_names = {k: v for k, v in self.feature_names.items()
                                  if v > 1}

        self.features = {k: idx for idx, k in enumerate(self.feature_names)}
        self.vec_len = len(self.feature_names)
        self._is_fit = True
        return self

    def vectorize(self, x):
        """Vectorize some data."""
        s = np.zeros(self.vec_len)
        if self.prune:
            indices = [self.features[s] for s in x if s in self.features]
        else:
            indices = [self.features[s] for s in x]
        s[indices] = 1
        return s


class HypernymSemanticsTransformer(BaseTransformer):
    """
    Create semantic vectors by one-hot encoding meronym relations.

    This is an implementation of the semantics approach from the
    Triangle model.
    For each word, it starts at the synset to which this word belongs, and
    moves up in the wordnet hierarchy, adding the parent nodes to the
    representation of that word at each step.
    The representation can also include meronyms, which possibly makes it more
    general.

    If you use it, please cite:
    @techreport{harm2002building,
      title={Building large scale distributed semantic
             feature sets with WordNet},
      author={Harm, MW},
      year={2002},
      institution={Technical Report PDP-CNS-02-1, Carnegie Mellon University}
    }

    Note that this transformer differs from the other transformers because it
    is only able to represent exactly the data on which it is fit. Trying to
    transform new words leads to errors.

    Parameters
    ----------
    field : str or None
        The field to retrieve.

    use_meronyms : bool
        Whether to also retrieve meronyms. Retrieving meronyms causes words to
        have more links to each other, as words which are not hypernyms can
        still share meronyms. This does have the effect of blowing up the
        dimensionality of the space.

    """

    def __init__(self, field=None, use_meronyms=True, prune=False):
        """Initialize the transformer."""
        super().__init__(field)
        self.use_meronyms = use_meronyms
        self.prune = prune

    def fit(self, X):
        """Fit the transformer."""
        super().fit(X)
        X = self._unpack(X)

        related = Counter()
        for x in X:
            for (offset, pos) in x:
                s = wn.synset_from_pos_and_offset(pos, int(offset))
                related.update(self.recursive_related(s))

        if self.prune:
            related = {k: v for k, v in related.items() if v > 1}
        self.feature_names = set(chain.from_iterable(X))
        self.features = {k: idx for idx, k in enumerate(related)}
        self.vec_len = len(self.features)
        self._is_fit = True
        return self

    def vectorize(self, x):
        """Vectorize a word."""
        vec = np.zeros(self.vec_len)
        for (offset, pos) in x:
            s = wn.synset_from_pos_and_offset(pos, int(offset))
            res = self.recursive_related(s)
            if self.prune:
                res = [x for x in res if x in self.features]
            s = [self.features[s] for s in res]
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
