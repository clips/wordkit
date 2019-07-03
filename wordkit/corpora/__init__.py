"""Corpora."""
from .corpora import (celex,
                      cmu,
                      subtlex,
                      lexique,
                      bpal,
                      wordnet,
                      lexiconproject)
from .base import reader, Frame

__all__ = ["celex",
           "cmu",
           "subtlex",
           "bpal",
           "lexique",
           "wordnet",
           "lexiconproject",
           "reader",
           "Frame"]
