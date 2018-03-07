"""Readers for corpora."""
from .celex import Celex
from .cmudict import CMU
from .deri import Deri
from .corpusaugmenter import CorpusAugmenter
from .subtlex import Subtlex
from .lexique import Lexique
from .bpal import BPal

__all__ = ["Celex",
           "CMU",
           "Deri",
           "CorpusAugmenter",
           "Subtlex",
           "BPal",
           "Lexique"]
