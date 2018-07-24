"""Read semantic information from multilingual wordnets."""
from .base import Reader


FIELD_IDS = {"semantics": 0, "orthography": 2, "language": -1}


class WordNet(Reader):
    """
    Reader for Wordnet tab-separated files.

    This reader can handle the TAB-files format from the following link:
    http://compling.hss.ntu.edu.sg/omw/

    """

    def __init__(self, path, language, fields=("orthography",
                                               "semantics",
                                               "language")):
        """Get semantic information."""
        super().__init__(path,
                         fields=fields,
                         field_ids=FIELD_IDS,
                         language=language,
                         merge_duplicates=True)

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
        use_s = 'semantics' in self.fields

        if wordlist:
            wordlist = set([x.lower() for x in wordlist])

        words_added = set()

        # path to phonology part of the CELEX database
        for line in iterable:

            line = line.strip()
            columns = line.split('\t')
            orthography = columns[self.field_ids['orthography']].lower()

            word = {}

            if wordlist and orthography not in wordlist:
                continue
            words_added.add(orthography)
            if use_o:
                word['orthography'] = orthography
            if use_s:
                word['semantics'] = columns[self.field_ids['semantics']]

            yield word
