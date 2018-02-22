"""Base class for readers."""
import os

from collections import defaultdict
from sklearn.base import TransformerMixin


def identity(x):
    """Identity function."""
    return x


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
    ==========
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
        If this is False, duplicates may occur in the output.
    filter_function : filter_function, optional, default identity
        A custom function you can use to filter the output.
        An example of this could be a frequency selection function.

    Examples
    ========
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
                 merge_duplicates=False,
                 filter_function=None):
        """Base class for readers."""
        if not os.path.exists(path):
            raise ValueError("The file you specified does not "
                             "exist: {}".format(path))

        difference = set(fields) - set(field_ids.keys())
        if difference:
            raise ValueError("You provided fields which are not valid "
                             "for this database-language pair: {}. "
                             "Valid features are "
                             "{}.".format(difference,
                                          set(field_ids.keys())))

        fields = {f: field_ids[f] for f in fields}
        self.path = path
        self.fields = dict(fields)
        self.field_ids = field_ids
        self.merge_duplicates = merge_duplicates
        self.language = language
        self.orthographyfield = field_ids['orthography']
        self.filter_function = filter_function

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
        ==========
        X : list of strings.
            The orthographic form of the input words.
        y: None
            For sklearn compatibility.

        Returns
        =======
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        words = self._retrieve(X, kwargs=kwargs)

        if 'language' in self.fields:
            for w in words:
                w.update({"language": self.language})

        if self.merge_duplicates:
            new_words = defaultdict(int)
            for w in words:
                it = tuple([i for i in w.items() if i[0] != 'frequency'])
                try:
                    new_words[it] += w['frequency']
                except KeyError:
                    pass

            words = []

            for k, v in new_words.items():
                d = dict(k)
                if 'frequency' in self.fields:
                    d['frequency'] = v
                words.append(d)
        else:
            new_words = {tuple(w.items()) for w in words}
            new_words = [dict(w) for w in words]

        # Break lazy evaluation because users expect lists.
        words = list(filter(self.filter_function, words))

        return words

    def _retrieve(self,
                  wordlist,
                  **kwargs):

        raise NotImplemented("Base class")
