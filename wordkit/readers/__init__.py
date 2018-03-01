"""Readers for corpora."""
from .celex import Celex
from .cmudict import CMU
from .deri import Deri
from .corpusaugmenter import CorpusAugmenter

__all__ = ["Celex", "CMU", "Deri", "CorpusAugmenter"]
