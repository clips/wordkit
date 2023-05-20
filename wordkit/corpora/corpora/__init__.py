"""Import the various corpora."""
from wordkit.corpora.corpora.bpal import bpal
from wordkit.corpora.corpora.celex import celex_dutch, celex_english, celex_german
from wordkit.corpora.corpora.cmudict import cmu
from wordkit.corpora.corpora.lexiconproject import blp, clp, dlp1, dlp2, elp, flp, klp
from wordkit.corpora.corpora.lexique import lexique
from wordkit.corpora.corpora.subtlex import subtlexnl, subtlexuk, subtlexus, subtlexzh
from wordkit.corpora.corpora.wordnet import wordnet

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
