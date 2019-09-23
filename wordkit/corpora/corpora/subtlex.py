"""Corpus readers for Subtlex."""
from ..base import reader

LANG2FIELD = {"eng-uk": {"orthography": "Spelling",
                         "frequency": "FreqCount"},
              "eng-us": {"orthography": "Word",
                         "frequency": "FREQcount"},
              "nld": {"orthography": "Word",
                      "frequency": "FREQcount"},
              "deu": {"orthography": "Word",
                      "frequency": "WFfreqCount"},
              "zh": {"orthography": "Word",
                     "frequency": "WCount"}}

LANG2SEP = {"eng-uk": ",",
            "nld": "\t",
            "zh": ",",
            "deu": ",",
            "eng-us": ","}


def subtlex(path,
            fields=("orthography", "frequency"),
            language=None):
    """Initialize the subtlex reader."""
    if language is None:
        raise ValueError("Please pass a language")
    if language == "chi":
        skiprows = 2
    else:
        skiprows = 0

    return reader(path,
                  fields,
                  LANG2FIELD[language],
                  language,
                  sep=LANG2SEP[language],
                  skiprows=skiprows)


def subtlexuk(path,
              fields=("orthography", "frequency")):
    return subtlex(path, fields, "eng-uk")


def subtlexus(path,
              fields=("orthography", "frequency")):
    return subtlex(path, fields, "eng-us")


def subtlexnl(path,
              fields=("orthography", "frequency")):
    return subtlex(path, fields, "eng-us")


def subtlexde(path,
              fields=("orthography", "frequency")):
    return subtlex(path, fields, "ger")


def subtlexzh(path,
              fields=("orthography", "frequency")):
    return subtlex(path, fields, "zh")
