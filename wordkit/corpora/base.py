"""Base class for corpus readers."""
import os
import re
import numpy as np
import pandas as pd
import random

from sklearn.base import TransformerMixin
from collections import defaultdict
from functools import partial

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


def apply_if_not_na(x, func):
    """Applies function to something if it is not NA."""
    return x if np.isnan(x) else func(x)


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


class BaseReader(TransformerMixin):
    """
    A reader that takes as input a dataframe instead of a file.
    """

    def __init__(self, data):
        """
        A reader that just takes data.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe containing the data this is supposed to process.

        """
        self.fields = list(data.columns)
        self.data = WordStore(data.to_dict('records'))

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
        return self.data.filter(filter_function, **kwargs)


class Reader(BaseReader):
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
                 duplicates,
                 scale_frequencies=True,
                 diacritics=diacritics,
                 **kwargs):
        """Init the base class."""
        if not os.path.exists(path):
            raise FileNotFoundError("The file you specified does not "
                                    "exist: {}".format(path))

        self.path = path
        self.language = language
        fields = {k: field_ids.get(k, k) for k in fields}
        self.duplicates = duplicates
        self.diacritics = diacritics
        self.scale_frequencies = scale_frequencies
        data = self._open(fields, **kwargs)
        super().__init__(data)

    def _open(self, fields, **kwargs):
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
            colnames = set(df.columns)
            redundant = set(fields.values()) - colnames
            if redundant:
                raise ValueError("You passed fields which were not in "
                                 "the dataset {}. The available fields are: "
                                 "{}".format(redundant, colnames))
        else:
            _, indices = zip(*fields.items())
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
                                 "- language: you supplied {}\n"
                                 "The original error was:\n"
                                 "'{}'".format(indices,
                                               sep,
                                               encoding,
                                               self.language,
                                               e))

        return self._preprocess(df, fields)

    def _preprocess(self, df, fields):
        """Preprocess the file."""
        keys, indices = zip(*fields.items())

        inverted = defaultdict(set)
        for k, v in fields.items():
            inverted[v].add(k)
        inverted = {k: v for k, v in inverted.items() if len(v) > 1}
        df = df.rename(columns=dict(zip(indices, keys)))
        for v in inverted.values():
            in_df = v & set(df.columns)
            in_df_name = list(in_df)[0]
            for field in v - in_df:
                df = df.assign(**{field: df[in_df_name]})
        df = df.loc[:, keys]

        # Drop nans before further processing.
        use_freq = 'frequency' in fields

        # Assign language, but we need to see whether this is a user-assigned
        # property or not.
        if 'language' in fields and self.language:
            df = df[df['language'] == self.language].copy()
        elif self.language:
            df['language'] = self.language

        if 'orthography' in fields:
            df['orthography'] = df['orthography'].astype(str)

        # Process phonology
        if 'phonology' in fields:
            func = partial(apply_if_not_na, func=self._process_phonology)
            df['phonology'] = df.apply(lambda x:
                                       func(x['phonology']),
                                       axis=1)
        # Process syllabic phonology
        if 'syllables' in fields:
            func = partial(apply_if_not_na, func=self._process_syllable)
            df['syllables'] = df.apply(lambda x:
                                       func(x['syllables']),
                                       axis=1)
        # Process semantics
        if 'semantics' in fields:
            func = partial(apply_if_not_na, func=self._process_semantics)
            df['semantics'] = df.apply(lambda x:
                                       func(x['semantics']),
                                       axis=1)
            # This might return NaNs
            df = df.dropna()
            other_fields = tuple(set(df.columns) - {'semantics'})
            g = df.groupby(other_fields)

            # Slow, but only way this works.
            df['semantics'] = g['semantics'].transform(np.sum)
            # Drop duplicate entries
            df = df.drop_duplicates().copy()

        if df.empty:
            raise ValueError("All your rows contained at least one NaN.")

        # We want to merge duplicates, but we don't want to merge on the
        # basis of frequency. Instead, we sum the frequency.
        if self.duplicates:
            ungroupable = {'frequency'}
            cols_to_group = list(set(df.columns) - ungroupable)
            if use_freq:
                g = df.groupby(cols_to_group)['frequency']
                if self.duplicates == "sum":
                    df.loc[:, ('frequency',)] = g.transform(np.sum)
                elif self.duplicates == "max":
                    df.loc[:, ('frequency',)] = g.transform(np.max)
            df = df.drop_duplicates().copy()

        # Scale the frequencies.
        if use_freq and self.scale_frequencies:
            mask = ~np.isnan(df.frequency)
            mask_freq = df.frequency[mask]
            summ = np.sum(mask_freq)
            total = np.sum(mask_freq) / 1e6
            m = min(mask_freq[mask_freq > 0])
            smoothed_total = (summ + (len(mask_freq) * m)) / 1e6
            d = np.zeros(len(df)) * np.nan
            d[mask] = mask_freq / total
            df['frequency_per_million'] = d
            d = np.zeros(len(df)) * np.nan
            d[mask] = np.log10(mask_freq + m)
            df['log_frequency'] = d
            # Should reference publication.
            d = np.zeros(len(df)) * np.nan
            d[mask] = np.log10((mask_freq + m) / smoothed_total)
            d[mask] += 3
            df['zipf_score'] = d

        return df

    def fit(self, X, y=None):
        """Static, no fit."""
        return self

    def _process_syllable(self, x):
        """identity function."""
        return x

    def _process_phonology(self, x):
        """identity function."""
        return x

    def _process_semantics(self, x):
        """identity function."""
        return x


class WordStore(list):
    """A wordstore class."""

    def __init__(self, *args, **kwargs):
        """Initialize the wordstore."""
        super().__init__(*args, **kwargs)
        # field to type mapping
        self.fields = defaultdict(set)
        for x in self:
            for k, v in x.items():
                self.fields[k].add(type(v))
        self.fields = {k: next(iter(v)) if len(v) == 1 else None
                       for k, v in self.fields.items()}

    def get(self, key, strict=False, na_value=None):
        """
        Gets values of a key from all words in the wordstore.

        Parameters
        ----------
        key : object
            The key to retrieve all items.
        strict : bool
            If this is True, raise a KeyError if a key is not present.
            If this is False, na_value is inserted.
        na_value : None or object
            The value to insert if a key is not present.

        Returns
        -------
        values : np.array
            The value of the key to retrieve.

        """
        X = []
        for x in self:
            try:
                X.append(x[key])
            except KeyError as e:
                if strict:
                    raise e
                X.append(na_value)
        return np.array(X)

    def filter(self, filter_function=None, **kwargs):
        """
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
                    if callable(v):
                        try:
                            t = v(x[k])
                        except KeyError:
                            # If something doesn't have the key, it does not
                            # get selected.
                            t = False
                    elif isinstance(v, self.fields[k]):
                        t = x[k] == v
                    elif isinstance(v, (tuple, set, list)):
                        t = x[k] in set(v)
                    else:
                        raise ValueError("We don't know what to do with the "
                                         "value you passed for the key {} "
                                         "".format(k))
                    if not t:
                        return False
            return True

        # Check which kwargs pertain to the data.
        functions = {k: v for k, v in kwargs.items()}
        # Only if we actually have functions should we do something.
        if functions:
            # If we also have a filter function, we should compose it
            if filter_function:
                functions['__general__'] = filter_function
            filter_function = partial(_filter, functions)
        return WordStore(filter(filter_function, self))

    def sample(self, n, distribution_key=None):
        """
        Sample from the wordstore.

        Parameters
        ----------
        n : int
            The number of items to sample.
        distribution_key : object, None, default None
            The key to use as a distribution. The values in this key needs to
            be an integer or a float for this to work.
            If this value is None, sampling is uniform

        Returns
        -------
        words : WordStore
            The sampled words.

        """
        if distribution_key is None:
            sample = [random.choice(self) for x in range(n)]
        else:
            distribution = np.array([x[distribution_key] for x in self])
            distribution = distribution / distribution.sum()
            sample = np.random.choice(self, size=n, p=distribution).tolist()
        return WordStore(sample)
