"""Corpora."""
from .corpora import (Celex,
                      CMU,
                      Deri,
                      Subtlex,
                      Lexique,
                      BPal,
                      WordNet,
                      LexiconProject)
from .base import Reader

__all__ = ["Celex",
           "CMU",
           "Deri",
           "merge",
           "Subtlex",
           "BPal",
           "Lexique",
           "WordNet",
           "LexiconProject",
           "Reader"]
