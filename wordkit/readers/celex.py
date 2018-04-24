"""Tools for working with Celex."""
import regex as re
import logging
import os

from .base import Reader, segment_phonology
from itertools import chain
from copy import copy

remove_double = re.compile(r"ː+")


logger = logging.getLogger(__name__)

AUTO_LANGUAGE = {"epl.cd": "eng",
                 "dpl.cd": "nld",
                 "gpl.cd": "deu",
                 "epw.cd": "eng",
                 "dpw.cd": "nld",
                 "gpw.cd": "deu"}

# MAX_FREQ has different content depending on whether we are using lemmas
# or words, so we index the MAX_FREQ dictionary using both the language
# and whether we are using lemmas.
MAX_FREQ = {('eng', True): 18632568,
            ('eng', False): 18740716,
            ('nld', True): 40370584,
            ('nld', False): 40181484,
            ('deu', True): 5054170,
            ('deu', False): 1000000}

# MAX_FREQ is scaled to 1M, which scales the corpus to 1M words.
MAX_FREQ = {k: v / 1000000 for k, v in MAX_FREQ.items()}

language2field = {'eng': {'orthography': 1,
                          'phonology': 7,
                          'frequency': 2,
                          'syllables': 7,
                          'language': None,
                          'log_frequency': None},
                  'nld': {'orthography': 1,
                          'phonology': 5,
                          'frequency': 2,
                          'syllables': 5,
                          'language': None,
                          'log_frequency': None},
                  'deu': {'orthography': 1,
                          'phonology': 4,
                          'frequency': 2,
                          'syllables': 4,
                          'language': None,
                          'log_frequency': None}}

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

        super().__init__(path,
                         fields,
                         p,
                         language,
                         merge_duplicates,
                         frequency_divider=MAX_FREQ[(language, self.lemmas)])

        self.replace = re.compile(r"(,|r\*)")
        self.braces = re.compile(r"[\[\]]+")
        self.double_braces = re.compile(r"(\[[^\]]+?)\[(.+?)\]([^\[])")

    def _retrieve(self, iterable, wordlist=None, **kwargs):
        """
        Extract word information from the CELEX database.

        Parameters
        ----------
        wordlist : list of strings or None.
            The list of words to be extracted from the corpus.
            If this is None, all words are extracted.

        Returns
        -------
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        use_o = 'orthography' in self.fields
        use_p = 'phonology' in self.fields
        use_syll = 'syllables' in self.fields
        use_freq = 'frequency' in self.fields
        use_log_freq = 'log_frequency' in self.fields

        if wordlist:
            wordlist = set([x.lower() for x in wordlist])

        words_added = set()

        # path to phonology part of the CELEX database
        for line in iterable:

            line = line.strip()
            columns = line.split('\\')
            orthography = columns[self.field_ids['orthography']].lower()

            word = {}

            if wordlist and orthography not in wordlist:
                continue
            words_added.add(orthography)
            if use_o:
                word['orthography'] = orthography
            if use_p or use_syll:
                try:
                    syll = columns[self.field_ids['phonology']]
                except KeyError:
                    syll = columns[self.field_ids['syllables']]
                if not syll:
                    logging.info("{} has no associated phonological or "
                                 "syllable info, skipping".format(orthography))
                    continue
                phon = syll
                if use_syll:
                    syll = self.double_braces.sub("\g<1>\g<2>][\g<2>\g<3>",
                                                  syll)
                    syll = [self.replace.sub("", x)
                            for x in self.braces.split(syll) if x]
                    syll = [segment_phonology(x) for x in celex_to_ipa(syll)]
                    word['syllables'] = tuple(syll)
                if use_p:
                    phon = [self.replace.sub("", x)
                            for x in self.braces.split(phon) if x]
                    phon = [segment_phonology(x) for x in celex_to_ipa(phon)]
                    word['phonology'] = tuple(chain.from_iterable(phon))
            if use_freq or use_log_freq:
                word['frequency'] = int(columns[self.field_ids['frequency']])
                word['frequency'] += 1

            yield word
