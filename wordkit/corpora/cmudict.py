"""Tools for working with CMUDICT."""
from .base import Reader


CMU_2IPA = {'AO': 'ɔ',
            'AO0': 'ɔ',
            'AO1': 'ɔ',
            'AO2': 'ɔ',
            'AA': 'ɑ',
            'AA0': 'ɑ',
            'AA1': 'ɑ',
            'AA2': 'ɑ',
            'IY': 'i',
            'IY0': 'i',
            'IY1': 'i',
            'IY2': 'i',
            'UW': 'u',
            'UW0': 'u',
            'UW1': 'u',
            'UW2': 'u',
            'EH': 'e',
            'EH0': 'e',
            'EH1': 'e',
            'EH2': 'e',
            'IH': 'ɪ',
            'IH0': 'ɪ',
            'IH1': 'ɪ',
            'IH2': 'ɪ',
            'UH': 'ʊ',
            'UH0': 'ʊ',
            'UH1': 'ʊ',
            'UH2': 'ʊ',
            'AH': 'ʌ',
            'AH0': 'ə',
            'AH1': 'ʌ',
            'AH2': 'ʌ',
            'AE': 'æ',
            'AE0': 'æ',
            'AE1': 'æ',
            'AE2': 'æ',
            'AX': 'ə',
            'AX0': 'ə',
            'AX1': 'ə',
            'AX2': 'ə',
            'EY': 'eɪ',
            'EY0': 'eɪ',
            'EY1': 'eɪ',
            'EY2': 'eɪ',
            'AY': 'aɪ',
            'AY0': 'aɪ',
            'AY1': 'aɪ',
            'AY2': 'aɪ',
            'OW': 'oʊ',
            'OW0': 'oʊ',
            'OW1': 'oʊ',
            'OW2': 'oʊ',
            'AW': 'aʊ',
            'AW0': 'aʊ',
            'AW1': 'aʊ',
            'AW2': 'aʊ',
            'OY': 'ɔɪ',
            'OY0': 'ɔɪ',
            'OY1': 'ɔɪ',
            'OY2': 'ɔɪ',
            'P': 'p',
            'B': 'b',
            'T': 't',
            'D': 'd',
            'K': 'k',
            'G': 'g',
            'CH': 'tʃ',
            'JH': 'dʒ',
            'F': 'f',
            'V': 'v',
            'TH': 'θ',
            'DH': 'ð',
            'S': 's',
            'Z': 'z',
            'SH': 'ʃ',
            'ZH': 'ʒ',
            'HH': 'h',
            'M': 'm',
            'N': 'n',
            'NG': 'ŋ',
            'L': 'l',
            'R': 'r',
            'ER': 'ɜr',
            'ER0': 'ɜr',
            'ER1': 'ɜr',
            'ER2': 'ɜr',
            'AXR': 'ər',
            'AXR0': 'ər',
            'AXR1': 'ər',
            'AXR2': 'ər',
            'W': 'w',
            'Y': 'j'}


def cmu_to_ipa(phonemes):
    """Convert CMU phonemes to IPA unicode format."""
    return tuple([CMU_2IPA[p] for p in phonemes])


class CMU(Reader):
    """
    The reader for the CMUDICT corpus.

    The CMUDICT corpus can be downloaded here:
        https://github.com/cmusphinx/cmudict

    Currently, the CMUDICT has no proper reference, so please refer to the
    github or webpage of CMU if you use this resource.

    Parameters
    ----------
    path : string
        The path to the corpus this reader has to read.

    language : string, default "eng"
        The language of the corpus.

    fields : iterable, default ("orthography", "phonology"")
        An iterable of strings containing the fields this reader has
        to read from the corpus.

    merge_duplicates : bool, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "phonology"),
                 language=None,
                 merge_duplicates=True):
        """Extract structured information from CMUDICT."""
        super().__init__(path,
                         fields,
                         {'orthography': 0,
                          'phonology': 1},
                         language="eng",
                         merge_duplicates=merge_duplicates)

    def _retrieve(self, iterable, wordlist=None, **kwargs):
        """
        Extract sequences of phonemes for each word from the CMUDICT database.

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

        wordlist = set([x.lower() for x in wordlist])
        words_added = set()

        for line in iterable:

            line = line.strip()
            columns = line.split()
            columns = columns[0], columns[1:]
            orthography = columns[self.field_ids['orthography']].lower()

            word = {}

            if wordlist and orthography not in wordlist:
                continue
            words_added.add(orthography)
            word['orthography'] = orthography
            if use_p:
                syll = cmu_to_ipa(columns[self.field_ids['phonology']])
                word['phonology'] = syll
            if 'language' in self.fields:
                word['language'] = self.language

            yield word
