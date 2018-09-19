"""Read the Lexique database."""
from .base import Reader, segment_phonology, diacritics
from itertools import chain


LEXIQUE_2IPA = {'°': 'ə',
                'a': 'a',
                'k': 'k',
                '§': 'ɔ̃',
                'p': 'p',
                'l': 'l',
                'i': 'i',
                'j': 'j',
                '@': 'ɑ̃',
                'O': 'ɔ',
                'R': 'ʁ',
                'E': 'ɛ',
                's': 's',
                'y': 'y',
                'o': 'o',
                'S': 'ʃ',
                'b': 'b',
                '1': 'œ̃',
                '2': 'ø',
                '5': 'ɛ̃',
                '8': 'ɥ',
                '9': 'œ',
                'G': 'ŋ',
                'N': 'ɲ',
                'Z': 'ʒ',
                'd': 'd',
                'e': 'e',
                'f': 'f',
                'g': 'ɡ',
                'm': 'm',
                'n': 'n',
                't': 't',
                'u': 'u',
                'v': 'v',
                'w': 'w',
                'x': 'x',
                'z': 'z'}


def lexique_to_ipa(syllables):
    """Convert Lexique phonemes to IPA unicode format."""
    for syll in syllables:
        yield "".join([LEXIQUE_2IPA[x] for x in syll])


class Lexique(Reader):
    """
    Read the Lexique corpus.

    This reader reads the Lexique corpus, which contains frequency,
    orthography, phonology and syllable fields for 125,733 French words.

    The Lexique corpus does not have an associated publication, and can be
    accessed at this link: http://www.lexique.org/

    Parameters
    ----------
    path : str
        The path to the Lexique corpus file.

    fields : tuple
        The fields to retrieve from the corpus.

    language : str
        The language of the corpus. Currently not used in Lexique.

    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        Note that frequency is not counted as a field for determining
        duplicates. Frequency is instead added together for any duplicates.
        If this is False, duplicates may occur in the output.

    """

    def __init__(self,
                 path,
                 fields=('orthography', 'phonology', 'frequency'),
                 language=None,
                 merge_duplicates=True,
                 diacritics=diacritics,
                 scale_frequencies=True):
        """Initialize the reader."""
        super().__init__(path,
                         fields,
                         {"orthography": "1_ortho",
                          "phonology": "23_syll",
                          "frequency": "10_freqlivres",
                          "syllables": "23_syll",
                          "log_frequency": "10_freqlivres"},
                         "fra",
                         merge_duplicates,
                         diacritics=diacritics,
                         scale_frequencies=scale_frequencies)
        self.data = self._open(sep="\t")

    def _process_phonology(self, string):
        """Process phonology."""
        string = string.split("-")
        string = tuple(lexique_to_ipa(string))
        string = [segment_phonology(x) for x in string]
        return tuple(chain.from_iterable(string))

    def _process_syllable(self, string):
        """Process syllables."""
        string = string.split("-")
        string = tuple(lexique_to_ipa(string))
        string = [segment_phonology(x) for x in string]
        return tuple(string)
