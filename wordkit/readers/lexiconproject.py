"""Readers for the lexicon projects."""
from .base import Reader


language2field = {"eng": {},
                  "br-eng": {},
                  "fra": {},
                  "zh": {},
                  "pol": {},
                  "dut": {}}

language2sep = {"eng": ",",
                "nld": "\t",
                "dut": "\t",
                "eng-uk": "\t"}


class LexiconProject(Reader):
    """Docstring."""

    def __init__(self, path, fields, language, merge_duplicates):

        field_ids = language2field[language]
        super().__init__(path, fields, field_ids, merge_duplicates)
        self.sep = language2sep[language]

    def _retrieve(self, iterable, wordlist=None):
        """
        Extract word information for each word from the database.

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
        use_orth = "orthography" in self.fields
        use_rt = "rt" in self.fields

        for line in iterable:

            word = {}
            columns = line.strip().split(self.sep)
            orthography = columns[self.field_ids['orthography']].lower()
            if wordlist and orthography not in wordlist:
                continue
            if use_orth:
                word['orthography'] = orthography
            if use_rt:
                rt = float(columns[self.field_ids['rt']].lower())
                word['rt'] = rt
            for x in self.fields:
                data = columns[self.field_ids[x]]
                try:
                    data = float(data)
                except ValueError:
                    pass
                word[x] = data

            yield word
