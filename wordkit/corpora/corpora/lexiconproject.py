"""Readers for lexicon project corpora."""
from ..base import reader

PROJECT2SEP = {"blp": "\t", "dlp2": "\t", "dlp1": "\t"}
PROJECT2FIELD = {
    "dlp1": {"orthography": "spelling"},
    "dlp2": {"orthography": "spelling", "rt": "rtC.mean"},
    "blp": {"orthography": "spelling"},
    "elp": {"orthography": "Word", "rt": "I_Mean_RT", "frequency": "SUBTLWF"},
    "flp": {"orthography": "item"},
    "clp": {"orthography": "Character", "rt": "RT"},
    "klp": {
        "orthography": "Stimuli",
        "frequency": "Freq",
        "lexicality": "Lexicality",
        "rt": "Stim_RT_M",
    },
}


def blp(path, fields):
    """British Lexicon Project."""
    return reader(
        path, fields, PROJECT2FIELD["blp"], language="eng-uk", sep=PROJECT2SEP["blp"]
    )


def elp(path, fields):
    """English Lexicon Project."""
    return reader(path, fields, PROJECT2FIELD["elp"], language="eng-us")


def flp(path, fields):
    """French Lexicon Project."""
    return reader(path, fields, PROJECT2FIELD["flp"], language="fra")


def dlp1(path, fields):
    """Dutch Lexicon Project 1."""
    return reader(
        path, fields, PROJECT2FIELD["dlp1"], language="nld", sep=PROJECT2SEP["dlp1"]
    )


def dlp2(path, fields):
    """Dutch Lexicon Project 2."""
    return reader(
        path, fields, PROJECT2FIELD["dlp2"], language="nld", sep=PROJECT2SEP["dlp2"]
    )


def clp(path, fields):
    """Chinese Lexicon Project."""
    return reader(path, fields, PROJECT2FIELD["clp"], language="zh")


def klp(path, fields):
    """Korean Lexicon Project."""
    return reader(path, fields, PROJECT2FIELD["klp"], language="kor")
