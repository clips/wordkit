"""The BPAL corpus reader."""
import re
from functools import partial
from itertools import chain

from wordkit.corpora.base import reader

BPAL_2IPA = {
    "TIO": "sjo",
    "#d": "d",
    "gl": "ɣr",
    "gr": "ɣr",
    "gj": "ɣi",
    "gw": "ɣw",
    "ga": "ɰa",
    "A": "a",
    "E": "e",
    "I": "i",
    "J": "tʃ",
    "L": "ʎ",
    "N": "ɲ",
    "O": "o",
    "R": "ɾ",
    "T": "s",
    "U": "u",
    "b": "β",
    "d": "ð",
    "f": "f",
    "g": "g",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "w": "w",
    "x": "x",
    "y": "ʝ",
}

bpal_regex = re.compile(r"{}".format("|".join(BPAL_2IPA.keys())))


def bpal_to_ipa(syllables):
    """Convert bpal phonemes to IPA unicode format."""
    for idx, x in enumerate(syllables):
        converted = "".join([BPAL_2IPA[p] for p in bpal_regex.findall(x)])
        if idx == 0:
            converted = converted.replace("ð", "d")
            converted = converted.replace("β", "b")
        yield tuple(converted)


def syll_func(string):
    """Process a BPAL syllable string."""
    string = string.split("-")
    string = tuple(bpal_to_ipa(string))

    return tuple(string)


def phon_func(string):
    """Process a BPAL phonology string."""
    string = string.split("-")
    string = tuple(bpal_to_ipa(string))

    return tuple(chain.from_iterable(string))


bpal = partial(
    reader,
    field_ids={"orthography": 0, "syllables": 1, "phonology": 1},
    language="esp",
    sep="\t",
    encoding="latin-1",
    quoting=0,
    header=None,
    preprocessors={"phonology": phon_func, "syllables": syll_func},
)
