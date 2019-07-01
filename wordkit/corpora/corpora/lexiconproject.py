"""Readers for lexicon project corpora."""
import os
from ..base import Reader


ALLOWED_LANGUAGES = ("eng-uk", "eng-us", "fra", "nld", "chi", "kor")
LANG_SEP = {"eng-uk": "\t", "nld": "\t", "eng-us": ","}
language2field = {"nld": {"orthography": "spelling"},
                  "eng-uk": {"orthography": "spelling"},
                  "eng-us": {"orthography": "Word", "rt": "I_Mean_RT"},
                  "fra": {"orthography": "item"},
                  "chi": {"orthography": "Character", "rt": "RT"},
                  "kor": {"orthography": "Stimuli",
                          "frequency": "Freq",
                          "lexicality": "Lexicality",
                          "rt": "Stim_RT_M"}}
AUTO_LANGUAGE = {"french lexicon project words.xls": "fra",
                 "blp-items.txt": "eng-uk",
                 "dlp_items.txt": "nld",
                 "elp-items.csv": "eng-us",
                 "Chinese Lexicon Project Sze et al.xlsx": "chi",
                 "klp_ld_item_ver.1.0.xlsx": "kor"}


class LexiconProject(Reader):
    """
    Lexicon projects are a set of corpora which contain Reaction Time
    measurements for large sets of words.

    All lexicon projects can be found here:
    http://crr.ugent.be/programs-data/lexicon-projects

    If you use a lexicon project, please cite the appropriate paper.

    Parameters
    ----------
    path : string
        The path to the corpus this reader has to read.
    fields : iterable
        An iterable of strings containing the fields this reader has
        to read from the corpus.
    language : string
        The language of the corpus.

    Example
    -------
    >>> from string import ascii_lowercase
    >>> def freq_alpha(x):
    >>>     a = set(x['orthography']) - set(ascii_lowercase)
    >>>     b = x['frequency'] > 10
    >>>     return (not a) and b
    >>>
    >>> r = Reader("/path/",
    >>>            ("orthography", "frequency"),
    >>>            "eng")
    >>> words = r.transform(filter_function=freq_alpha)

    """
    def __init__(self,
                 path,
                 fields=("orthography", "rt"),
                 language=None):
        """Initialize the reader."""
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
        if language not in language2field:
            langs = set(language2field.keys())
            raise ValueError("Your language {}, was not in the set of "
                             "allowed languages: {}".format(language,
                                                            langs))
        super().__init__(path,
                         fields,
                         language2field[language],
                         language,
                         sep=LANG_SEP.get(language, None))
