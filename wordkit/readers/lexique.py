"""Read the Lexique database."""
from .base import Reader, segment_phonology, diacritics
from itertools import chain

max_freq = 1963616 / 1000000


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
                 diacritics=diacritics):
        """Initialize the reader."""
        super().__init__(path,
                         fields,
                         {"orthography": 0,
                          "phonology": 22,
                          "frequency": [8, 9],
                          "syllables": 22,
                          "log_frequency": None},
                         "fra",
                         merge_duplicates,
                         frequency_divider=max_freq,
                         diacritics=diacritics)

    def _retrieve(self, iterable, wordlist, **kwargs):
        """
        Extract word information for each word from the databases.

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
        # skip header
        next(iterable)

        use_o = 'orthography' in self.fields
        use_p = 'phonology' in self.fields
        use_syll = 'syllables' in self.fields
        use_freq = 'frequency' in self.fields
        use_log_freq = 'log_frequency' in self.fields

        if wordlist:
            wordlist = set([x.lower() for x in wordlist])

        words_added = set()

        for idx, line in enumerate(iterable):
            columns = line.strip().split("\t")
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
                    try:
                        syll = columns[self.field_ids['syllables']]
                    except IndexError:
                        print(columns, len(columns), idx)

                if use_syll or use_p:
                    syll = syll.split("-")
                    syll = tuple(lexique_to_ipa(syll))
                    syll = [segment_phonology(x) for x in syll]
                    if use_p:
                        word['phonology'] = tuple(chain.from_iterable(syll))
                    if use_syll:
                        word['syllables'] = tuple(syll)
            if use_freq or use_log_freq:
                freq = sum([float(columns[x])
                            for x in self.field_ids["frequency"]])
                word['frequency'] = freq

            yield word
