"""Base class for readers."""
import os
import regex as re
import numpy as np

from collections import defaultdict
from sklearn.base import TransformerMixin


remove_double = re.compile(r"(ː)(\1){1,}")

diacritics = {'ː',
              '̤',
              'ˠ',
              '̠',
              '̈',
              '̞',
              '̩',
              '̻',
              'ʰ',
              'ʼ',
              '̝',
              'ʲ',
              '̥',
              '̟',
              'ˤ',
              '̃',
              '̺',
              '͡',
              '̯',
              '̪',
              '̰',
              'ʷ'}


def identity(x):
    """Identity function."""
    return x


def segment_phonology(phonemes, items=diacritics, to_keep=diacritics):
    """
    Segment a list of characters into chunks by joining diacritics.

    This function turns a string of phonemes, which might have diacritics and
    suprasegmentals as separate characters, into a list of phonemes in which
    any diacritics or suprasegmentals have been joined.

    It also removes any diacritics which are not in the list to_keep,
    allowing the user to systematically remove any distinction which is not
    deemed useful for the current research.

    Additionally, any double longness markers are removed, and replaced by
    a single marker.

    Parameters
    ----------
    phonemes : list
        A list of phoneme characters to segment.
    items : list
        A list of characters which to treat as diacritics.
    to_keep : list
        A list of diacritics from the list passed to items which to keep.
        Any items in this list are not removed as spurious diacritics.
        If to_keep and items are the same list, all items are kept.

    """
    phonemes = remove_double.sub("\g<1>", phonemes)
    phonemes = [list(p) for p in phonemes]
    idx = 0
    while idx < len(phonemes):
        x = phonemes[idx]
        if x[0] in items:
            if x[0] in to_keep:
                phonemes[idx-1].append(x[0])
            phonemes.pop(idx)
        else:
            idx += 1

    return tuple(["".join(x) for x in phonemes if x])


class Reader(TransformerMixin):
    """
    Base class for corpus readers.

    In Wordkit, a corpus reader is intended for reading structured corpora,
    e.g. Celex, which contain extra information associated with words.

    Typically, this information is the phonology, frequency or syllable
    structure associated with a word. Each source of information is
    specified as a string called a "field". Fields serve 2 purposes: they
    delineate which information can be found in a corpus, and they serve as
    an index to the information in a corpus.

    Parameters
    ----------
    path : string
        The path to the corpus this reader has to read.
    fields : iterable
        An iterable of strings containing the fields this reader has
        to read from the corpus.
    language : string
        The language of the corpus.
    merge_duplicates : bool, optional, default False
        Whether to merge duplicates which are indistinguishable according
        to the selected fields.
        Note that frequency is not counted as a field for determining
        duplicates. Frequency is instead added together for any duplicates.
        If this is False, duplicates may occur in the output.
    filter_function : filter_function, optional, default identity
        A custom function you can use to filter the output.
        An example of this could be a frequency selection function.
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
    >>>            "eng",
    >>>            filter_function=freq_alpha)
    >>> words = r.transform([])

    """

    def __init__(self,
                 path,
                 fields,
                 field_ids,
                 language,
                 merge_duplicates,
                 filter_function,
                 diacritics=diacritics,
                 frequency_divider=1):
        """Init the base class."""
        if not os.path.exists(path):
            raise FileNotFoundError("The file you specified does not "
                                    "exist: {}".format(path))

        difference = set(fields) - set(field_ids.keys())
        if difference:
            raise ValueError("You provided fields which are not valid "
                             "for this database-language pair: {}. "
                             "Valid features are "
                             "{}.".format(difference,
                                          set(field_ids.keys())))

        self.path = path
        self.fields = {f: field_ids[f] for f in fields}
        self.field_ids = field_ids
        self.merge_duplicates = merge_duplicates
        self.language = language
        self.orthographyfield = field_ids['orthography']
        self.filter_function = filter_function
        self.diacritics = diacritics
        self.frequency_divider = frequency_divider

    def fit(self, X, y=None):
        """Static, no fit."""
        return self

    def transform(self,
                  X=(),
                  y=None,
                  **kwargs):
        """
        Transform a list of words into dictionaries.

        This function reads all words in X from the corpus it points
        to, and retrieves structured information associated with that word.
        The type of structured information retrieved depends on the
        fields, which are given at initialization time.

        Parameters
        ----------
        X : list of strings.
            The orthographic form of the input words.
        y: None
            For sklearn compatibility.

        Returns
        -------
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        words = list(self._retrieve(X, kwargs=kwargs))

        if 'language' in self.fields:
            # Only add language if the transformer has not added it.
            for w in (x for x in words if 'language' not in x):
                w.update({"language": self.language})

        # Merging duplicates means that any duplicates are removed
        # and their frequencies are added together.
        if self.merge_duplicates:
            new_words = defaultdict(int)
            for w in words:
                it = tuple([i for i in w.items() if i[0] != "frequency"])
                try:
                    new_words[it] += w['frequency'] + 1
                except KeyError:
                    pass

            words = []

            if 'log_frequency' in self.fields:
                max_log_freq = np.log10(self.frequency_divider)

            for k, v in new_words.items():
                d = dict(k)
                if 'frequency' in self.fields:
                    d['frequency'] = v / self.frequency_divider
                if 'log_frequency' in self.fields:
                    # Note to reader:
                    # this looks wrong, because you're not supposed to use
                    # division in log space.
                    # In this case, however, we're trying to maintain
                    # the distances between logs.
                    # that is, we want each logged frequency to have the same
                    # proportional distance to each other logged frequency as
                    # before.
                    d['log_frequency'] = np.log10(v) / max_log_freq
                words.append(d)

        return list(filter(self.filter_function, words))

    def _retrieve(self,
                  wordlist,
                  **kwargs):

        raise NotImplemented("Base class")
