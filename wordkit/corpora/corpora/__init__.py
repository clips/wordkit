"""Import the various corpora."""
from .celex import Celex
from .cmudict import CMU
from .deri import Deri
from .subtlex import Subtlex
from .lexique import Lexique
from .bpal import BPal
from .wordnet import WordNet
from .lexiconproject import LexiconProject

__all__ = ["Celex",
           "CMU",
           "Deri",
           "merge",
           "Subtlex",
           "BPal",
           "Lexique",
           "WordNet",
           "LexiconProject"]
