"""Base class for corpus readers."""
import os
import regex as re
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from collections import defaultdict
from functools import partial

# special collection of nans because words like nan and null do occur in our
# corpora. These do not.
nans = {'',
        '#N/A',
        '#N/A N/A',
        '#NA',
        '-1.#IND',
        '-1.#QNAN',
        '-NaN',
        '-nan',
        '1.#IND',
        '1.#QNAN',
        'N/A',
        'NA',
        'NULL',
        'NaN'}

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
    Base class for corpora readers.

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
                 field_ids,
                 language,
                 merge_duplicates,
                 scale_frequencies=False,
                 diacritics=diacritics,
                 **kwargs):
        """Init the base class."""
        if not os.path.exists(path):
            raise FileNotFoundError("The file you specified does not "
                                    "exist: {}".format(path))

        self.path = path
        self.language = language
        self.fields = {k: field_ids.get(k, k) for k in fields}
        self.merge_duplicates = merge_duplicates
        self.diacritics = diacritics
        self.scale_frequencies = scale_frequencies
        self.data = self._open(**kwargs)

    def _open(self, **kwargs):
        """Open a file for reading."""
        header = kwargs.get('header', "infer")
        sep = kwargs.get('sep', ",")
        quoting = kwargs.get('quote', 0)
        encoding = kwargs.get('encoding', 'utf-8')
        comment = kwargs.get('comment', None)
        skiprows = kwargs.get('skiprows', None)

        extension = os.path.splitext(self.path)[-1]
        if extension in {".xls", ".xlsx"}:
            df = pd.read_excel(self.path,
                               skiprows=skiprows,
                               na_values=nans,
                               keep_default_na=False)
        else:
            fields, indices = zip(*self.fields.items())
            try:
                df = pd.read_csv(self.path,
                                 sep=sep,
                                 usecols=indices,
                                 quoting=quoting,
                                 header=header,
                                 encoding=encoding,
                                 keep_default_na=False,
                                 comment=comment,
                                 na_values=nans)
            except ValueError as e:
                raise ValueError("Something went wrong during reading of "
                                 "your data. Things that could be wrong: \n"
                                 "- column names: you supplied {}\n"
                                 "- separator: you supplied {}\n"
                                 "- encoding: you supplied {}\n"
                                 "The original error was:\n"
                                 "'{}'".format(indices, sep, encoding, e))

        return self._preprocess(df)

    def _preprocess(self, df):
        """Preprocess the file."""
        fields, indices = zip(*self.fields.items())

        inverted = defaultdict(set)
        for k, v in self.fields.items():
            inverted[v].add(k)
        inverted = {k: v for k, v in inverted.items() if len(v) > 1}
        df = df.rename(columns=dict(zip(indices, fields)))
        for v in inverted.values():
            in_df = v & set(df.columns)
            in_df_name = list(in_df)[0]
            for field in v - in_df:
                df = df.assign(**{field: df[in_df_name]})
        df = df.loc[:, fields]

        if 'language' in self.fields and self.language:
            df = df[df['language'] == self.language].copy()
        elif self.language:
            df['language'] = self.language

        if 'orthography' in self.fields:
            df['orthography'] = df['orthography'].astype(str)
            df['orthography'] = df.apply(lambda x: x['orthography'].lower(),
                                         axis=1)
        if 'phonology' in self.fields:
            df['phonology'] = df.apply(lambda x:
                                       self._process_phonology(x['phonology']),
                                       axis=1)
        if 'syllables' in self.fields:
            df['syllables'] = df.apply(lambda x:
                                       self._process_syllable(x['syllables']),
                                       axis=1)
        if 'semantics' in self.fields:
            df['semantics'] = df.apply(lambda x:
                                       self._process_semantics(x['semantics']),
                                       axis=1)
            # This might return NaNs
            df = df.dropna()
            other_fields = tuple(set(df.columns) - {'semantics'})
            g = df.groupby(other_fields)

            df['semantics'] = g['semantics'].transform(np.sum)
            df = df.drop_duplicates().copy()

        use_freq = 'frequency' in self.fields

        df = df.dropna()
        if df.empty:
            raise ValueError("All your rows contained at least one NaN.")

        if self.merge_duplicates:
            ungroupable = {'frequency'}
            cols_to_group = list(set(df.columns) - ungroupable)
            if use_freq:
                g = df.groupby(cols_to_group)['frequency']
                df.loc[:, ('frequency',)] = g.transform(np.sum)
            df = df.drop_duplicates().copy()

        if use_freq and self.scale_frequencies:
            summ = np.sum(df.frequency)
            total = np.sum(df.frequency) / 1e6
            smoothed_total = (summ + len(df.frequency)) / 1e6
            df['frequency_per_million'] = df['frequency'] / total
            df['log_frequency'] = np.log10(df['frequency'] + 1)
            df['zipf_score'] = np.log10((df['frequency'] + 1) / smoothed_total)
            df['zipf_score'] += 3

        return df

    def fit(self, X, y=None):
        """Static, no fit."""
        return self

    def transform(self,
                  X=(),
                  y=None,
                  filter_function=None,
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
        y : None
            For sklearn compatibility.
        filter_function : function or None, default None
            The filtering function to use. A filtering function is a function
            which accepts a dictionary as argument and which returns a boolean
            value. If the filtering function returns False, the item is not
            retrieved from the corpus.

            Example of a filtering function could be a function which
            constrains the frequencies of retrieved words, or the number of
            syllables.
        kwargs : lambda function
            This function also takes general keyword arguments that take
            keys as keys and functions as values. This offers a more flexible
            alternative to the filter_function option above.

            e.g. if "frequency" is a field, you can use
                frequency=lambda x: x > 10
            as a keyword argument to only retrieve items with a frequency > 10.

        Returns
        -------
        words : list of dictionaries
            Each entry in the dictionary represents the structured information
            associated with each word. This list need not be the length of the
            input list, as words can be expressed in multiple ways.

        """
        words = self.data.to_dict('records')
        # Kwargs contains functions
        # compose a new function on the fly using _filter

        def _filter(functions, x):
            """
            Generate new filter_function

            This is a composition of boolean functions which are chained
            with AND statements. Hence, if any of the functions evaluates to
            False we can return False without evaluating all of them.

            This is better than using any([v(x) for v in functions]) because
            there we evaluate all functions before calling any([]).

            """

            if not functions:
                return True
            for k, v in functions.items():
                if k == '__general__':
                    t = v(x)
                else:
                    t = v(x[k])
                if not t:
                    return False
            return True

        # Check which kwargs pertain to the data.
        functions = {k: v for k, v in kwargs.items() if k in self.fields}
        # Only if we actually have functions should we do something.
        if functions:
            # If we also have a filter function, we should compose it
            if filter_function:
                functions['__general__'] = filter_function
            filter_function = partial(_filter, functions)
        if X:
            wordlist = set(X)
            words = [x for x in words if x['orthography'] in wordlist]
        return list(filter(filter_function, words))

    def get_sampler(self,
                    num_to_sample,
                    filter_function=None,
                    replacement=False,
                    max_iter=10000,
                    **kwargs):
        """
        Returns a Sampler that samples from the corpus.

        If you want to sample using the frequencies of the words, please use
        the Sampler classes from wordkit.sampler.

        Parameters
        ----------
        num_to_sample : int
            The number of words to sample.
        filter_function : function
            The filtering function to use. A filtering function is a function
            which accepts a dictionary as argument and which returns a boolean
            value. If the filtering function returns False, the item is not
            retrieved from the corpus.

            Example of a filtering function could be a function which
            constrains the frequencies of retrieved words, or the number of
            syllables.
        max_iter : int
            The maximum number of iterations this generator is useable.
            This is just a safeguard because we don't want computers to crash
            just because someone coerces this generator to a list.

        Returns
        -------
        sampler : generator
            An infinite generator that returns words.

        """
        words = self.transform(filter_function=filter_function, **kwargs)
        if len(words) <= num_to_sample:
            raise ValueError("num_to_sample is equal or larger than the "
                             "number of words in your corpus. {} > {}"
                             "".format(num_to_sample, len(words)))

        return ([words[idx] for idx in np.random.choice(len(words),
                                                        size=num_to_sample,
                                                        replace=False)]
                for x in range(max_iter))

    def _process_syllable(self, x):
        """identity function."""
        return x

    def _process_phonology(self, x):
        """identity function."""
        return x

    def _process_semantics(self, x):
        """identity function."""
        return x
