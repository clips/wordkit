"""Readers for corpora."""
from .celex import Celex
from .cmudict import CMU
from .deri import Deri
from .merge import merge
from .subtlex import Subtlex
from .lexique import Lexique
from .bpal import BPal

__all__ = ["Celex",
           "CMU",
           "Deri",
           "merge",
           "Subtlex",
           "BPal",
           "Lexique"]
