"""Tools for working with the multilingual corpora from Deri and Knight."""
import re

from .base import Reader, identity
from ipapy.ipastring import IPAString

language2field = {'orthography': 2, 'phonology': 3}


class Deri(Reader):
    r"""
    The reader for the Deri and Knight series of corpuses.

    These corpora are described in the paper:
    Grapheme-to-Phoneme Models for (Almost) Any Language
    by Deri and Knight (2016).

    This reader is different from the others because all the languages
    are in a single file. We still allow the user to only specify a single
    language per reader.

    The set of corpora can be downloaded here:
        https://drive.google.com/drive/folders/0B7R_gATfZJ2aSlJabDMweU14TzA

    If you use this corpus reader or the corpora, you _must_ cite the following
    paper:

    @inproceedings{deri2016grapheme,
      title={Grapheme-to-Phoneme Models for (Almost) Any Language.},
      author={Deri, Aliya and Knight, Kevin}
    }

    Parameters
    ==========
    path : string
        The path to the corpus this reader has to read.
    fields : iterable, optional, default ("orthography", "phonology")
        An iterable of strings containing the fields this reader has
        to read from the corpus.
    language : string, optional, default ("eng")
        The language of the corpus.
    min_freq : int, optional, default None
        Words with a frequency below this value are discarded.
    max_freq : int, optional, default None
        Words with a frequency above this value are discarded.
    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        If this is False, duplicates may occur in the output.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "phonology"),
                 language='eng',
                 merge_duplicates=False,
                 filter_function=identity):
        """Extract words from Deri and Knight corpora."""
        super().__init__(path,
                         fields,
                         language2field,
                         merge_duplicates,
                         filter_function)

        self.language = language
        self.matcher = re.compile(r"([:/]|rhymes)")

    def _retrieve(self, wordlist=None, **kwargs):
        """
        Extract sequences of phonemes for each word from the databases.

        Parameters
        ==========
        wordlist : list of strings or None.
            The list of words to be extracted from the corpus.
            If this is None, all words are extracted.

        Returns
        =======
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        use_p = 'phonology' in self.fields

        wordlist = set([x.lower() for x in wordlist])
        result = []
        words_added = set()

        for line in open(self.path):

            line = line.strip()
            columns = line.split("\t")
            if columns[0] not in self.language:
                continue
            orthography = columns[self.orthographyfield].lower()

            out = {}

            if wordlist and orthography not in wordlist:
                continue
            m = self.matcher.finditer(orthography)
            try:
                next(m)
                continue
            except StopIteration:
                pass
            words_added.add(orthography)
            out['orthography'] = "_".join(orthography.split())
            if use_p:
                syll = columns[self.fields['phonology']].split()
                syll = "".join(syll)
                syll = "".join([str(x) for x in IPAString(unicode_string=syll)
                                if not x.is_suprasegmental])

                out['phonology'] = syll
            if 'language' in self.fields:
                out['language'] = self.language

            result.append(out)

        return result
