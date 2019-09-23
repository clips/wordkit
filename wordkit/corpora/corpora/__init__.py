"""Import the various corpora."""
from .celex import celex_german, celex_english, celex_dutch
from .cmudict import cmu
from .subtlex import subtlexnl, subtlexuk, subtlexus, subtlexzh
from .lexique import lexique
from .bpal import bpal
from .wordnet import wordnet
from .lexiconproject import elp, blp, flp, dlp1, dlp2, klp, clp

__all__ = ["celex_english",
           "celex_dutch",
           "celex_german",
           "cmu",
           "subtlex",
           "bpal",
           "lexique",
           "wordnet",
           "elp",
           "blp",
           "flp",
           "dlp1",
           "dlp2",
           "klp",
           "clp",
           "subtlexnl",
           "subtlexuk",
           "subtlexus",
           "subtlexzh"]
