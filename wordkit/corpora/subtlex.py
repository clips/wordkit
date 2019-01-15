"""Corpus readers for Subtlex."""
import os
from .base import Reader

# Currently redundant, but useful for future-proofing.
language2field = {"eng-uk": {"orthography": "Spelling",
                             "frequency": "FreqCount"},
                  "eng-us": {"orthography": "Word",
                             "frequency": "FREQcount"},
                  "nld": {"orthography": "Word",
                          "frequency": "FREQcount"},
                  "deu": {"orthography": "Word",
                          "frequency": "WFfreqCount"},
                  "chi": {"orthography": "Word",
                          "frequency": "WCount"}}

ALLOWED_LANGUAGES = set(language2field.keys())
AUTO_LANGUAGE = {'subtlex-ch-wf.xlsx': 'chi',
                 'subtlex-de cleaned with google00 frequencies.xlsx': 'deu',
                 'subtlex-nl.cd-above2.txt': 'nld',
                 'subtlex-uk.xlsx': 'eng-uk',
                 'subtlexusfrequencyabove1.xls': 'eng-us'}


class Subtlex(Reader):
    """
    Reader for the various Subtlex corpora.

    These are corpora of frequency norms which explain significantly more
    variance than other frequency norms based on large corpora.

    In general, all the Subtlex corpora are associated with a paper.
    For an overview, visit:
        http://crr.ugent.be/programs-data/subtitle-frequencies

    Please make sure to read the associated articles, and cite them whenever
    you use a version of Subtlex!

    This class can read both the Excel and csv versions of the subtlex files.
    If you can, please consider converting the Excel files offered by the
    original authors to a csv before processing as this will significantly
    speed up the entire process.

    Parameters
    ----------
    path : string
        The path to the file. The extension of this file will be used to
        determine the method we use to open the file.
    fields : tuple, default ("orthography", "frequency")
        The fields to retrieve from this corpus.
    language : string, default "eng-uk"
        The language the corpus is in.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "frequency"),
                 language=None,
                 merge_duplicates=True,
                 scale_frequencies=True):
        """Initialize the subtlex reader."""
        if language is None:
            try:
                language = AUTO_LANGUAGE[os.path.split(path)[1].lower()]
            except KeyError:
                raise ValueError("You passed None to language, but we failed "
                                 "to determine the language automatically.")
        else:
            try:
                if AUTO_LANGUAGE[os.path.split(path)[1]] != language:
                    raise ValueError("Your language is {}, but your filename "
                                     "belongs to another language."
                                     "".format(language))
            except KeyError:
                pass
        if language not in ALLOWED_LANGUAGES:
            raise ValueError("Your language {}, was not in the set of "
                             "allowed languages: {}".format(language,
                                                            ALLOWED_LANGUAGES))
        if language == "chi":
            skiprows = 2
        else:
            skiprows = 0
        super().__init__(path,
                         fields,
                         language2field[language],
                         language,
                         merge_duplicates=merge_duplicates,
                         diacritics=None,
                         scale_frequencies=scale_frequencies,
                         sep="\t",
                         skiprows=skiprows)
