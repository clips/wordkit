"""Tools for working with Celex."""
import re
import os
import csv

from ..base import reader, segment_phonology
from itertools import chain
from copy import copy
from functools import partial


remove_double = re.compile(r"ː+")


AUTO_LANGUAGE = {"epl.cd": "eng-uk",
                 "dpl.cd": "nld",
                 "gpl.cd": "deu",
                 "epw.cd": "eng-uk",
                 "dpw.cd": "nld",
                 "gpw.cd": "deu"}

language2field = {'eng-uk': {'orthography': 1,
                             'phonology': 7,
                             'frequency': 2,
                             'syllables': 7},
                  'nld': {'orthography': 1,
                          'phonology': 5,
                          'frequency': 2,
                          'syllables': 5},
                  'deu': {'orthography': 1,
                          'phonology': 4,
                          'frequency': 2,
                          'syllables': 4}}

lengths = {("nld", True): (11, 0),
           ("eng-uk", True): (4, 4),
           ("deu", True): (11, 0),
           ("nld", False): (7, 0),
           ("eng-uk", False): (5, 4),
           ("deu", False): (7, 0)}

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
          lemmas=None):
    """Extract structured information from CELEX."""
    if language is None:
        try:
            language = AUTO_LANGUAGE[os.path.split(path)[1].lower()]
        except KeyError:
            raise ValueError("You passed None to language, but we failed "
                             "to determine the language automatically.")
    else:
        try:
            if AUTO_LANGUAGE[os.path.split(path)[1]] != language:
                raise ValueError("Your language is {}, but your filename "
                                 "belongs to another language."
                                 "".format(language))
        except KeyError:
            pass

    if lemmas is None:
        if path.endswith("l.cd"):
            lemmas = True
        elif path.endswith("w.cd"):
            lemmas = False
        else:
            raise ValueError("You passed None to lemmas, but we failed "
                             "to determine wether your files contained "
                             "lemmas automatically.")

    p = copy(language2field[language])
    if not lemmas:
        p['phonology'] += 1
        p['syllables'] += 1
    fields = {f: p.get(f, f) for f in fields}
    w_length, s_length = lengths[(language, lemmas)]
    _opener = partial(_celex_opener,
                      word_length=w_length,
                      struct_length=s_length)

    return reader(path,
                  fields,
                  p,
                  language,
                  delimiter="\\",
                  quoting=csv.QUOTE_NONE,
                  opener=_opener,
                  preprocessors={"phonology": phon_func,
                                 "syllables": syll_func})
