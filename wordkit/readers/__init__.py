"""Readers for corpora."""
from .celex import Celex
from .cmudict import CMU
from .deri import Deri
from .corpusaugmenter import CorpusAugmenter
from .subtlex import Subtlex

__all__ = ["Celex", "CMU", "Deri", "CorpusAugmenter", "Subtlex"]
