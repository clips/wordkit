"""Import the various corpora."""
from .celex import celex
from .cmudict import cmu
from .subtlex import subtlex
from .lexique import lexique
from .bpal import bpal
from .wordnet import wordnet
from .lexiconproject import lexiconproject

__all__ = ["celex",
           "cmu",
           "subtlex",
           "bpal",
           "lexique",
           "wordnet",
           "lexiconproject"]
