"""Tools for working with Celex."""
import csv
import re
from functools import partial
from itertools import chain

import pandas as pd

from wordkit.corpora.base import reader, segment_phonology

remove_double = re.compile(r"ː+")


PROJECT2FIELD = {
    ("eng", False): {"orthography": 1, "phonology": 8, "frequency": 2, "syllables": 8},
    ("nld", False): {"orthography": 1, "phonology": 6, "frequency": 2, "syllables": 6},
    ("deu", False): {"orthography": 1, "phonology": 5, "frequency": 2, "syllables": 5},
    ("eng", True): {"orthography": 1, "phonology": 7, "frequency": 2, "syllables": 7},
    ("nld", True): {"orthography": 1, "phonology": 5, "frequency": 2, "syllables": 5},
    ("deu", True): {"orthography": 1, "phonology": 4, "frequency": 2, "syllables": 4},
}

LENGTHS = {
    ("nld", True): (11, 0),
    ("eng", True): (4, 4),
    ("deu", True): (11, 0),
    ("nld", False): (7, 0),
    ("eng", False): (5, 4),
    ("deu", False): (7, 0),
}

CELEX_2IPA = {
    "O~": "ɒ̃",
    "A~": "ɒ",
    "&~": "æ",
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "g": "ɡ",
    "N": "ŋ",
    "m": "m",
    "n": "n",
    "l": "l",
    "r": "r",
    "f": "f",
    "v": "v",
    "T": "θ",
    "D": "ð",
    "s": "s",
    "z": "z",
    "S": "ʃ",
    "Z": "ʒ",
    "j": "j",
    "x": "x",
    "h": "h",
    "w": "w",
    "I": "ɪ",
    "E": "ɛ",
    "&": "æ",
    "V": "ʌ",
    "Q": "ɒ",
    "U": "ʊ",
    "@": "ə",
    "O": "ɔ",
    "3": "ɜ",
    "A": "ɑ",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
    "G": "x",
    "y": "y",
    ":": "ː",
}

CELEX_REGEX = re.compile(r"{}".format("|".join(CELEX_2IPA.keys())))
REPLACE = re.compile(r"(,|r\*)")
BRACES = re.compile(r"[\[\]]+")
DOUBLE_BRACES = re.compile(r"(\[[^\]]+?)\[(.+?)\]([^\[])")


def syll_func(string):
    """Process a CELEX syllable string."""
    syll = DOUBLE_BRACES.sub(r"\g<1>\g<2>][\g<2>\g<3>", string)
    syll = [REPLACE.sub("", x) for x in BRACES.split(syll) if x]
    syll = [segment_phonology(x) for x in celex_to_ipa(syll)]
    return tuple(syll)


def phon_func(string):
    """Process a CELEX phonology string."""
    phon = [REPLACE.sub("", x) for x in BRACES.split(string) if x]
    phon = [segment_phonology(x) for x in celex_to_ipa(phon)]
    return tuple(chain.from_iterable(phon))


def celex_to_ipa(syllables):
    """Convert celex phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([CELEX_2IPA[p] for p in CELEX_REGEX.findall(syll)])


def _celex_opener(path, word_length, struct_length=0, **kwargs):
    """Open a CELEX file for reading."""
    csv_file = csv.reader(open(path), **kwargs)
    data = []
    for line in csv_file:
        rem = len(line) - word_length
        if rem != 0:
            if not struct_length:
                raise ValueError(line)
            if rem % struct_length:
                raise ValueError(line)

        inform = line[:word_length]
        if struct_length == 0:
            data.append(dict(enumerate(inform)))
            continue
        for x in range(word_length, len(line), struct_length):
            data.append(dict(enumerate(inform + line[x : x + struct_length])))

    return pd.DataFrame(data)


def _celex(path, fields, lemmas, language):
    w_length, s_length = LENGTHS[(language, lemmas)]
    _opener = partial(_celex_opener, word_length=w_length, struct_length=s_length)

    return reader(
        path,
        fields,
        PROJECT2FIELD[(language, lemmas)],
        language,
        delimiter="\\",
        quoting=csv.QUOTE_NONE,
        opener=_opener,
        preprocessors={"phonology": phon_func, "syllables": syll_func},
    )


def celex_english(path, fields=("orthography", "syllables", "frequency"), lemmas=False):
    return _celex(path, fields, lemmas, "eng")


def celex_dutch(path, fields=("orthography", "syllables", "frequency"), lemmas=False):
    return _celex(path, fields, lemmas, "nld")


def celex_german(path, fields=("orthography", "syllables", "frequency"), lemmas=False):
    return _celex(path, fields, lemmas, "deu")
