"""Tools for working with Celex."""
import regex as re
import logging
import os

from .base import Reader, segment_phonology
from itertools import chain
from copy import copy
from csv import QUOTE_NONE

remove_double = re.compile(r"ː+")


logger = logging.getLogger(__name__)

AUTO_LANGUAGE = {"epl.cd": "eng",
                 "dpl.cd": "nld",
                 "gpl.cd": "deu",
                 "epw.cd": "eng",
                 "dpw.cd": "nld",
                 "gpw.cd": "deu"}

language2field = {'eng': {'orthography': 1,
                          'phonology': 7,
                          'frequency': 2,
                          'syllables': 7,
                          'log_frequency': 2},
                  'nld': {'orthography': 1,
                          'phonology': 5,
                          'frequency': 2,
                          'syllables': 5,
                          'log_frequency': 2},
                  'deu': {'orthography': 1,
                          'phonology': 4,
                          'frequency': 2,
                          'syllables': 4,
                          'log_frequency': 2}}

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


def celex_to_ipa(syllables):
    """Convert celex phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([CELEX_2IPA[p] for p in celex_regex.findall(syll)])


class Celex(Reader):
    r"""
    The reader for the CELEX corpus.

    This reader is built with the assumption that you have access to the lemma
    files of the Celex corpus. Normally these named like "xpl.cd", where x is
    a letter which refers to a language. More information can be found in the
    CELEX readme.

    Currently, we include readers for the Dutch and English parts of CELEX,
    but adding other readers should be straightforward and can be done
    by adding other translation dictionaries.

    If you use the CELEX corpus, you _must_ cite the following paper:

    @article{baayen1993celex,
      title={The $\{$CELEX$\}$ lexical data base on $\{$CD-ROM$\}$},
      author={Baayen, R Harald and Piepenbrock, Richard and van H, Rijn},
      year={1993},
      publisher={Linguistic Data Consortium}
    }

    Parameters
    ----------
    path : string
        The path to the corpus this reader has to read.

    language : string, default ("eng")
        The language of the corpus.

    fields : iterable, default ("orthography", "syllables", "frequency")
        An iterable of strings containing the fields this reader has
        to read from the corpus.

    merge_duplicates : bool, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "syllables", "frequency", "language"),
                 language=None,
                 merge_duplicates=True,
                 scale_frequencies=True,
                 lemmas=None):
        """Extract structured information from CELEX."""
        if not os.path.exists(path):
            raise FileNotFoundError("{} not found.".format(path))
        if language is None:
            try:
                language = AUTO_LANGUAGE[os.path.split(path)[1]]
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
                self.lemmas = True
            elif path.endswith("w.cd"):
                self.lemmas = False
            else:
                raise ValueError("You passed None to lemmas, but we failed "
                                 "to determine wether your files contained "
                                 "lemmas automatically.")

        p = copy(language2field[language])
        if not self.lemmas:
            p['phonology'] += 1
            p['syllables'] += 1
        fields = {k: v for k, v in p.items() if k in fields}

        self.replace = re.compile(r"(,|r\*)")
        self.braces = re.compile(r"[\[\]]+")
        self.double_braces = re.compile(r"(\[[^\]]+?)\[(.+?)\]([^\[])")

        super().__init__(path,
                         fields,
                         language2field[language],
                         language,
                         merge_duplicates,
                         scale_frequencies)
        self.data = self._open(sep="\\", quote=QUOTE_NONE, header=None)

    def _process_syllable(self, string):
        """Process a CELEX syllable string."""
        syll = self.double_braces.sub("\g<1>\g<2>][\g<2>\g<3>",
                                      string)
        syll = [self.replace.sub("", x)
                for x in self.braces.split(syll) if x]
        syll = [segment_phonology(x) for x in celex_to_ipa(syll)]
        return tuple(syll)

    def _process_phonology(self, string):
        """Process a CELEX phonology string."""
        phon = [self.replace.sub("", x)
                for x in self.braces.split(string) if x]
        phon = [segment_phonology(x) for x in celex_to_ipa(phon)]
        return tuple(chain.from_iterable(phon))
