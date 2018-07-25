"""Read semantic information from multilingual wordnets."""
from .base import Reader
from collections import defaultdict


FIELD_IDS = {"semantics": 0, "orthography": 2, "language": -1}


class WordNet(Reader):
    """
    Reader for Wordnet tab-separated files.

    This reader can handle the TAB-files format from the following link:
    http://compling.hss.ntu.edu.sg/omw/

    If you use any of these, make sure to cite the appropriate source, as well
    as the official WordNet reference:

    @article{miller1995wordnet,
      title={WordNet: a lexical database for English},
      author={Miller, George A},
      journal={Communications of the ACM},
      volume={38},
      number={11},
      pages={39--41},
      year={1995},
      publisher={ACM}
    }

    Parameters
    ----------
    path : str
        The path to the corpus.

    language : str
        The language of the corpus. This is not checked, so make sure that you
        put the appropriate language here.

    restrict_pos : list or None, default None
        If this is set to None, synsets of all Parts of Speech are accepted.
        If the list is non-empty, only synsets with parts of speech in the list
        are retrieved.
        The possible parts of speech in wordnet are {'n', 'a', 'v', 's'}.

    fields : tuple
        The fields to retrieve.

    """

    def __init__(self,
                 path,
                 language,
                 restrict_pos=None,
                 fields=("orthography",
                         "semantics",
                         "language")):
        """Get semantic information."""
        super().__init__(path,
                         fields=fields,
                         field_ids=FIELD_IDS,
                         language=language,
                         merge_duplicates=True)

        self.restrict_pos = restrict_pos

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
        use_s = 'semantics' in self.fields

        if wordlist:
            wordlist = set([x.lower() for x in wordlist])

        words_added = set()

        words = defaultdict(list)

        # path to phonology part of the CELEX database
        for line in iterable:

            if line.startswith("#"):
                continue

            line = line.strip()
            columns = line.split('\t')
            orthography = columns[self.field_ids['orthography']].lower()

            if wordlist and orthography not in wordlist:
                continue
            words_added.add(orthography)
            if use_s:
                offset, pos = columns[self.field_ids['semantics']].split("-")
                offset = int(offset)
                if self.restrict_pos is not None:
                    if pos in self.restrict_pos:
                        words[orthography].append((offset, pos))
                else:
                    words[orthography].append((offset, pos))

        for k, v in words.items():
            yield {'orthography': k, 'semantics': tuple(v)}
