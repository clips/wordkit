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
                 fields=("orthography", "syllables", "language", "phonology"),
                 language=None,
                 merge_duplicates=True):
        """Initialize the BPAL reader."""
        super().__init__(path,
                         fields,
                         {"orthography": 0,
                          "syllables": 1,
                          "language": None,
                          "phonology": 1},
                         "esp",
                         merge_duplicates,
                         diacritics=diacritics)

    def _open(self):
        """Open a file for reading."""
        return open(self.path, encoding="latin-1")

    def _retrieve(self, iterable, wordlist, *args, **kwargs):
        """
        Extract word information from the BPAL database.

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
        use_syll = "syllables" in self.fields
        use_phon = "phonology" in self.fields
        use_ortho = "orthography" in self.fields

        wordlist = set(wordlist)

        for line in iterable:

            columns = line.strip().split("\t")
            if len(columns) == 1:
                continue

            word = {}
            w = columns[self.field_ids['orthography']]

            if wordlist and w not in wordlist:
                continue

            if use_ortho:
                word['orthography'] = w

            if use_syll or use_phon:
                try:
                    syll = columns[self.field_ids['syllables']]
                except KeyError:
                    syll = columns[self.field_ids['phonology']]

                syll = syll.split("-")
                syll = tuple(bpal_to_ipa(syll))

                if use_syll:
                    word['syllables'] = tuple(syll)
                if use_phon:
                    word['phonology'] = tuple(chain.from_iterable(syll))

            yield word
