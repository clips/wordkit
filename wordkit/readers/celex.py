"""Tools for working with Celex."""
import re
import logging

from .base import Reader, identity, segment_phonology
from itertools import chain
from copy import copy

remove_double = re.compile(r"ː+")


logger = logging.getLogger(__name__)

language2field = {'eng': {'orthography': 1,
                          'phonology': 7,
                          'frequency': 2,
                          'syllables': 7,
                          'language': None,
                          'disc': 5},
                  'nld': {'orthography': 1,
                          'phonology': 5,
                          'frequency': 2,
                          'syllables': 5,
                          'language': None,
                          'disc': 3},
                  'ger': {'orthography': 1,
                          'phonology': 4,
                          'frequency': 2,
                          'syllables': 4,
                          'language': None,
                          'disc': 3}}


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


def celex_to_ipa(phonemes):
    """Convert celex phonemes to IPA unicode format."""
    return "".join([CELEX_2IPA[p] for p in celex_regex.findall(phonemes)])


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
    fields : iterable, default ("orthography", "syllables", "frequency")
        An iterable of strings containing the fields this reader has
        to read from the corpus.
    language : string, optional, default ("eng")
        The language of the corpus.
    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.
    translate_phonemes : bool, optional, default True
        Whether to translate phonemes using the CELEX_2IPA translation table.
        If this is set to False, the transformer will not be compatible with
        downstream phonological featurizers deliverd with Wordkit.
    filter_function : function
        The filtering function to use. A filtering function is a function
        which accepts a dictionary as argument and which returns a boolean
        value. If the filtering function returns False, the item is not
        retrieved from the corpus.

        Example of a filtering function could be a function which constrains
        the frequencies of retrieved words, or the number of syllables.
    disc_mode : bool, optional, default False
        Whether to get disc phonemes from CELEX. If this is set to True,
        translate phonemes is automatically set to False.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "syllables", "frequency", "language"),
                 language='eng',
                 merge_duplicates=False,
                 translate_phonemes=True,
                 filter_function=identity,
                 disc_mode=False):
        """Extract structured information from CELEX."""
        p = copy(language2field[language])

        if disc_mode:
            p['syllables'] = p['disc']
            p['phonology'] = p['disc']
            translate_phonemes = False

        super().__init__(path,
                         fields,
                         p,
                         language,
                         merge_duplicates,
                         filter_function)

        if disc_mode:
            self.replace = re.compile(r"'")
            self.braces = re.compile(r"-")
        else:
            self.replace = re.compile(r"(,|r\*)")
            self.braces = re.compile(r"[\[\]]+")
            self.double_braces = re.compile(r"(\[[^\]]+)\[(.+)\]([^\[])")
        self.translate_phonemes = translate_phonemes
        self.disc_mode = disc_mode

    def _retrieve(self, wordlist=None, **kwargs):
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
        use_p = 'phonology' in self.fields
        use_syll = 'syllables' in self.fields
        use_freq = 'frequency' in self.fields

        if wordlist:
            wordlist = set([x.lower() for x in wordlist])
        result = []
        words_added = set()

        # path to phonology part of the CELEX database
        for line in open(self.path):

            line = line.strip()
            columns = line.split('\\')
            orthography = columns[self.orthographyfield].lower()

            out = {}

            if wordlist and orthography not in wordlist:
                continue
            words_added.add(orthography)
            out['orthography'] = orthography
            if use_p or use_syll:
                try:
                    syll = columns[self.fields['phonology']]
                except KeyError:
                    syll = columns[self.fields['syllables']]
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
                    if self.translate_phonemes:
                        syll = [celex_to_ipa(x) for x in syll]
                    syll = [segment_phonology(x) for x in syll]
                    out['syllables'] = tuple(syll)
                if use_p:
                    phon = [self.replace.sub("", x)
                            for x in self.braces.split(phon) if x]
                    if self.translate_phonemes:
                        phon = [celex_to_ipa(x) for x in phon]
                    phon = [segment_phonology(x) for x in phon]
                    out['phonology'] = tuple(chain.from_iterable(phon))
            if use_freq:
                out['frequency'] = int(columns[self.fields['frequency']])
            result.append(out)

        return result
