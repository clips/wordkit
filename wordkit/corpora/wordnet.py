"""Read semantic information from multilingual wordnets."""
from .base import Reader


class WordNet(Reader):
    """
    Reader for Wordnet tab-separated files.

    This reader can handle the TAB-files format from the following link:
    http://compling.hss.ntu.edu.sg/omw/

    If you use any of these, make sure to cite the appropriate source, as well
    as the official WordNet reference:

    @book{_Fellbaum:1998,
      booktitle =	 "{WordNet}: An Electronic Lexical Database",
      address =	 "Cambridge, MA",
      editor =	 "Fellbaum, Christiane",
      publisher =	 "MIT Press",
      year =	 1998,
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
                         "semantics")):
        """Get semantic information."""
        super().__init__(path,
                         fields=fields,
                         field_ids={"semantics": 0,
                                    "orthography": 2},
                         language=language,
                         merge_duplicates=True)

        self.restrict_pos = restrict_pos
        self.data = self._open(comment="#",
                               sep="\t",
                               header=None)

    def _process_semantics(self, string):
        """Process semantics."""
        offset, pos = string.split("-")
        if self.restrict_pos and pos not in self.restrict_pos:
            return None
        else:
            return ((offset, pos),)
