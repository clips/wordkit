"""Import the various corpora."""
from .bpal import bpal
from .celex import celex_dutch, celex_english, celex_german
from .cmudict import cmu
from .lexiconproject import blp, clp, dlp1, dlp2, elp, flp, klp
from .lexique import lexique
from .subtlex import subtlexnl, subtlexuk, subtlexus, subtlexzh
from .wordnet import wordnet

__all__ = [
    "celex_english",
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
    "subtlexzh",
]
