"""Corpus readers for Subtlex."""
from ..base import reader
from ..base.utils import _calc_hash

PROJECT2FIELD = {"subtlex-uk": {"orthography": "Spelling",
                                "frequency": "FreqCount"},
                 "subtlex-us": {"orthography": "Word",
                                "frequency": "FREQcount"},
                 "subtlex-nl": {"orthography": "Word",
                                "frequency": "FREQcount"},
                 "subtlex-de": {"orthography": "Word",
                                "frequency": "WFfreqCount"},
                 "subtlex-ch": {"orthography": "Word",
                                "frequency": "WCount"}}

AUTO_LANGUAGE = {'subtlex-ch': 'chi',
                 'subtlex-de': 'deu',
                 'subtlex-nl': 'nld',
                 'subtlex-uk': 'eng-uk',
                 'subtlex-us': 'eng-us'}

HASHES = {'3eb1877de350ef9b59b82923ae88345d': 'subtlex-ch',
          '786e0d7d03f9ce93b26577b543c9eb0f': 'subtlex-nl',
          '95a270d4812047f1c02affbea3e23e28': 'subtlex-de',
          'af7302fac01340bb8a533240ff850016': 'subtlex-us',
          'e600f1c7067b65d068a1b8d7baf2d8d7': 'subtlex-uk'}


def subtlex(path,
            fields=("orthography", "frequency"),
            language=None,
            project=None):
    """Initialize the subtlex reader."""
    if project is None:
        project = HASHES[_calc_hash(path)]
    else:
        if project not in PROJECT2FIELD:
            raise ValueError("Your project is not correct. Allowed "
                             f"projects are {set(PROJECT2FIELD.keys())}")
    if language is None:
        try:
            language = AUTO_LANGUAGE[project]
        except KeyError:
            raise ValueError("You passed None to language, but we failed "
                             "to determine the language automatically.")

    if language == "chi":
        skiprows = 2
    else:
        skiprows = 0

    return reader(path,
                  fields,
                  PROJECT2FIELD[project],
                  language,
                  sep="\t",
                  skiprows=skiprows)
