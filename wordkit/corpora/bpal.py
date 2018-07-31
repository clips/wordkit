"""The BPAL corpus reader."""
import regex as re
from .base import Reader, diacritics
from itertools import chain


BPAL_2IPA = {"TIO": "sjo",
             "#d": 'd',
             'gl': 'ɣr',
             'gr': 'ɣr',
             'gj': 'ɣi',
             'gw': 'ɣw',
             'ga': 'ɰa',
             'A': 'a',
             'E': 'e',
             'I': 'i',
             'J': 'tʃ',
             'L': 'ʎ',
             'N': 'ɲ',
             'O': 'o',
             'R': 'ɾ',
             'T': 's',
             'U': 'u',
             'b': 'β',
             'd': 'ð',
             'f': 'f',
             'g': 'g',
             'j': 'j',
             'k': 'k',
             'l': 'l',
             'm': 'm',
             'n': 'n',
             'p': 'p',
             'r': 'r',
             's': 's',
             't': 't',
             'w': 'w',
             'x': 'x',
             'y': 'ʝ'}

bpal_regex = re.compile(r"{}".format("|".join(BPAL_2IPA.keys())))


def bpal_to_ipa(syllables):
    """Convert bpal phonemes to IPA unicode format."""
    for idx, x in enumerate(syllables):
        converted = "".join([BPAL_2IPA[p]
                             for p in bpal_regex.findall(x)])
        if idx == 0:
            converted = converted.replace("ð", "d")
            converted = converted.replace("β", "b")
        yield tuple(converted)


class BPal(Reader):
    r"""
    Corpus reader for the BPal corpus.

    The BPal corpus is included in the distribution of the BuscaPalabras
    tool (BPal) by Davis and Perea.

    This corpus reader reads the "nwphono.txt" file which is included in
    this distribution. We convert the phonemes of the BPal corpus to IPA to
    ensure cross-language compatibility.

    The tool can be downloaded at the following URL:
    http://www.pc.rhul.ac.uk/staff/c.davis/Utilities/B-Pal.zip

    If you use this corpus reader, you must cite:

    @article{davis2005buscapalabras,
      title={BuscaPalabras: A program for deriving orthographic and
             phonological neighborhood statistics and other psycholinguistic
             indices in Spanish},
      author={Davis, Colin J and Perea, Manuel},
      journal={Behavior Research Methods},
      volume={37},
      number={4},
      pages={665--671},
      year={2005},
      publisher={Springer}
    }

    Parameters
    ----------
    path : str
        The path to the nwphono.txt file.

    fields : tuple
        The fields to extract using this corpus reader. Any invalid fields
        will cause the reader to throw a ValueError.

    language : str
        This language field is here for compatibility, but is not used.

    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "syllables", "phonology"),
                 merge_duplicates=True,
                 scale_frequencies=True):
        """Initialize the BPAL reader."""
        allowed_fields = {"orthography": 0,
                          "syllables": 1,
                          "phonology": 1}

        super().__init__(path,
                         fields,
                         allowed_fields,
                         language="esp",
                         merge_duplicates=merge_duplicates,
                         diacritics=diacritics,
                         scale_frequencies=scale_frequencies)
        self.data = self._open(sep="\t",
                               encoding="latin-1",
                               quote=0,
                               header=None)

    def _process_syllable(self, string):
        """Process a CELEX syllable string."""
        string = string.split("-")
        string = tuple(bpal_to_ipa(string))

        return tuple(string)

    def _process_phonology(self, string):
        """Process a CELEX phonology string."""
        string = string.split("-")
        string = tuple(bpal_to_ipa(string))

        return tuple(chain.from_iterable(string))
