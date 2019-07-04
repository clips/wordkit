"""Corpus readers for Subtlex."""
import os
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

PROJECT2SEP = {"subtlex-uk": ",",
               "subtlex-nl": "\t",
               "subtlex-ch": ",",
               "subtlex-de": ",",
               "subtlex-us": ","}

AUTO_LANGUAGE = {'subtlex-ch': 'chi',
                 'subtlex-de': 'deu',
                 'subtlex-nl': 'nld',
                 'subtlex-uk': 'eng-uk',
                 'subtlex-us': 'eng-us'}

AUTO_PROJECT = {"SUBTLEX-CH-WF.xlsx": "subtlex-chi",
                "SUBTLEX-UK.xlsx": "subtlex-uk",
                "SUBTLEX-NL.cd-above2.txt": "subtlex-nl",
                "SUBTLEX-DE cleaned with Google00 frequencies.xlsx": "subtlex-de", # noqa
                "SUBTLEXusfrequencyabove1.xls": "subtlex-us"}


HASHES = {'786e0d7d03f9ce93b26577b543c9eb0f': 'subtlex-nl',
          '79c8672068fcf9b02e4080862c9eb17b': 'subtlex-ch',
          'b528ba6be0785aecda7ac893caf1e722': 'subtlex-us',
          'd171a587c9355ffb7ff5ed2ddfe55e89': 'subtlex-de',
          'f252d192d592812f9e5bd0b4a68fffdf': 'subtlex-uk'}


def subtlex(path,
            fields=("orthography", "frequency"),
            language=None,
            project=None):
    """Initialize the subtlex reader."""
    if project is None:
        try:
            hash = _calc_hash(path)
            project = HASHES[hash]
        except KeyError:
            try:
                project = AUTO_PROJECT[os.path.split(path)[-1]]
            except KeyError:
                raise ValueError("Your project is not correct. Allowed "
                                 f"projects are {set(PROJECT2FIELD.keys())}")
    else:
        try:
            project = AUTO_PROJECT[os.path.split(path)[-1]]
        except KeyError:
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
                  sep=PROJECT2SEP[project],
                  skiprows=skiprows)
