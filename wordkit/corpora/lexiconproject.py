"""Readers for lexicon project corpora."""
from .base import Reader


ALLOWED_LANGUAGES = ("eng-uk", "eng-us", "fra", "nld")
LANG_SEP = {"eng-uk": "\t", "nld": "\t", "fra": None, "eng-us": "\t"}
language2field = {"nld": {"orthography": "spelling"},
                  "eng-uk": {"orthography": "spelling"},
                  "eng-us": {"orthography": "Word", "rt": "I_Mean_RT"},
                  "fra": {"orthography": "item"}}


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

    field_ids : dict
        A mapping which maps the field names from your data to the internal
        names used by wordkit.

    language : string
        The language of the corpus.

    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        Note that frequency is not counted as a field for determining
        duplicates. Frequency is instead added together for any duplicates.
        If this is False, duplicates may occur in the output.

    scale_frequencies : bool, default False
        Whether to scale the frequencies by a pre-defined amount.

    diacritics : tuple
        The diacritic markers from the IPA alphabet to keep. All diacritics
        which are IPA valid can be correctly parsed by wordkit, but it may
        not be desirable to actually have them in the dataset.

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
                 fields,
                 language,
                 scale_frequencies=False):
        """Initialize the reader."""
        if language not in ALLOWED_LANGUAGES:
            raise ValueError("Your language {}, was not in the set of "
                             "allowed languages: {}".format(language,
                                                            ALLOWED_LANGUAGES))
        super().__init__(path,
                         fields,
                         language2field[language],
                         language,
                         merge_duplicates=False)
        self.data = self._open(sep=LANG_SEP[language])
