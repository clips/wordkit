"""Read the Lexique database."""
from functools import partial
from itertools import chain

from wordkit.corpora.base import reader, segment_phonology

LEXIQUE_2IPA = {
    "°": "ə",
    "a": "a",
    "k": "k",
    "§": "ɔ̃",
    "p": "p",
    "l": "l",
    "i": "i",
    "j": "j",
    "@": "ɑ̃",
    "O": "ɔ",
    "R": "ʁ",
    "E": "ɛ",
    "s": "s",
    "y": "y",
    "o": "o",
    "S": "ʃ",
    "b": "b",
    "1": "œ̃",
    "2": "ø",
    "5": "ɛ̃",
    "8": "ɥ",
    "9": "œ",
    "G": "ŋ",
    "N": "ɲ",
    "Z": "ʒ",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "m": "m",
    "n": "n",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "z": "z",
}


def lexique_to_ipa(syllables):
    """Convert Lexique phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([LEXIQUE_2IPA[x] for x in syll])


def phon_func(string):
    """Process phonology."""
    string = string.split("-")
    string = tuple(lexique_to_ipa(string))
    string = [segment_phonology(x) for x in string]
    return tuple(chain.from_iterable(string))


def syll_func(string):
    """Process syllables."""
    string = string.split("-")
    string = tuple(lexique_to_ipa(string))
    string = [segment_phonology(x) for x in string]
    return tuple(string)


lexique = partial(
    reader,
    field_ids={
        "orthography": "1_ortho",
        "phonology": "23_syll",
        "frequency": "10_freqlivres",
        "syllables": "23_syll",
    },
    language="fra",
    sep="\t",
    preprocessors={"phonology": phon_func, "syllables": syll_func},
)
