"""Tools for working with Celex."""
import re
import csv

from ..base import reader, segment_phonology
from ..base.utils import _calc_hash
from itertools import chain
from functools import partial


remove_double = re.compile(r"ː+")


PROJECT2LANGUAGE = {"epl.cd": "eng-uk",
                    "dpl.cd": "nld",
                    "gpl.cd": "deu",
                    "epw.cd": "eng-uk",
                    "dpw.cd": "nld",
                    "gpw.cd": "deu"}

HASHES = {'094d5bf93446fd4c31cb145af7f8bdf4': 'gpw.cd',
          '1f035f9e7fd19955c0f93e5e17f12126': 'epw.cd',
          '564782c12374346d62b6672b8f1b1002': 'epl.cd',
          '5cb4c4dd8a3651f3993b857d1c1b8d12': 'dpl.cd',
          '7e289712b00184c1e0d30dbb55a5a5fd': 'gpl.cd',
          'f7a3ad8feb80e58177fcd470ea27065d': 'dpw.cd'}

PROJECT2FIELD = {'epw.cd': {'orthography': 1,
                            'phonology': 7,
                            'frequency': 2,
                            'syllables': 7},
                 'dpw.cd': {'orthography': 1,
                            'phonology': 5,
                            'frequency': 2,
                            'syllables': 5},
                 'gpw.cd': {'orthography': 1,
                            'phonology': 4,
                            'frequency': 2,
                            'syllables': 4},
                 'epl.cd': {'orthography': 1,
                            'phonology': 8,
                            'frequency': 2,
                            'syllables': 8},
                 'dpl.cd': {'orthography': 1,
                            'phonology': 6,
                            'frequency': 2,
                            'syllables': 6},
                 'gpl.cd': {'orthography': 1,
                            'phonology': 5,
                            'frequency': 2,
                            'syllables': 5}}

lengths = {"dpl.cd": (11, 0),
           "epl.cd": (4, 4),
           "gpl.cd": (11, 0),
           "dpw.cd": (7, 0),
           "epw.cd": (5, 4),
           "gpw.cd": (7, 0)}

CELEX_2IPA = {"O~": "ɒ̃",
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
              ":": "ː"}

celex_regex = re.compile(r"{}".format("|".join(CELEX_2IPA.keys())))
replace = re.compile(r"(,|r\*)")
braces = re.compile(r"[\[\]]+")
double_braces = re.compile(r"(\[[^\]]+?)\[(.+?)\]([^\[])")


def syll_func(string):
    """Process a CELEX syllable string."""
    syll = double_braces.sub(r"\g<1>\g<2>][\g<2>\g<3>",
                             string)
    syll = [replace.sub("", x)
            for x in braces.split(syll) if x]
    syll = [segment_phonology(x) for x in celex_to_ipa(syll)]
    return tuple(syll)


def phon_func(string):
    """Process a CELEX phonology string."""
    phon = [replace.sub("", x)
            for x in braces.split(string) if x]
    phon = [segment_phonology(x) for x in celex_to_ipa(phon)]
    return tuple(chain.from_iterable(phon))


def celex_to_ipa(syllables):
    """Convert celex phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([CELEX_2IPA[p] for p in celex_regex.findall(syll)])


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
            data.append(dict(enumerate(inform + line[x:x+struct_length])))

    return data


def celex(path,
          fields=("orthography", "syllables", "frequency", "language"),
          language=None,
          lemmas=None,
          project=None):
    """Extract structured information from CELEX."""
    if project is None:
        hash = _calc_hash(path)
        project = HASHES[hash]
    else:
        if project not in PROJECT2FIELD:
            raise ValueError("Your project is not correct. Allowed "
                             f"projects are {set(PROJECT2FIELD.keys())}")
    if language is None:
        try:
            language = PROJECT2LANGUAGE[project]
        except KeyError:
            language = None

    w_length, s_length = lengths[project]
    _opener = partial(_celex_opener,
                      word_length=w_length,
                      struct_length=s_length)

    return reader(path,
                  fields,
                  PROJECT2FIELD[project],
                  language,
                  delimiter="\\",
                  quoting=csv.QUOTE_NONE,
                  opener=_opener,
                  preprocessors={"phonology": phon_func,
                                 "syllables": syll_func})
