"""Base class for corpus readers."""
import os
import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import chain


nans = {'#N/A',
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


def _open(path, **kwargs):
    """Open a file for reading."""
    extension = os.path.splitext(path)[-1]
    if extension in {".xls", ".xlsx"}:
        df = pd.read_excel(path,
                           na_values=nans,
                           keep_default_na=False,
                           **kwargs)
    else:
        try:
            df = pd.read_csv(path,
                             na_values=nans,
                             keep_default_na=False,
                             engine="python",
                             **kwargs)
        except ValueError as e:
            sep = kwargs.get("sep", ",")
            encoding = kwargs.get("encoding", "utf-8")
            raise ValueError("Something went wrong during reading of "
                             "your data. Things that could be wrong: \n"
                             f"- separator: you supplied {sep}\n"
                             f"- encoding: you supplied {encoding}\n"
                             f"The original error was: {e}")

    return df


def reader(path,
           fields=None,
           field_ids=None,
           language=None,
           preprocessors=None,
           opener=_open,
           **kwargs):
    """Init the base class."""
    if not os.path.exists(path):
        raise FileNotFoundError("The file you specified does not "
                                f"exist: {path}")
    if isinstance(fields, str):
        fields = (fields,)
    if fields is None:
        fields = tuple()
    if field_ids is None:
        field_ids = {}

    df = opener(path, **kwargs)
    # Columns in dataset
    colnames = set(df.columns)

    rev = defaultdict(list)
    for k, v in field_ids.items():
        rev[v].append(k)
    c = set(chain.from_iterable([rev.get(x, [x]) for x in colnames]))
    redundant = set(fields) - c
    if redundant:
        raise ValueError("You passed fields which were not in "
                         f"the dataset {redundant}. The available fields "
                         f"are: {c}")
    if not fields:
        fields = c
    fields = {k: field_ids.get(k, k) for k in fields}

    for k, v in ((k, v) for k, v in fields.items() if k != v):
        df[k] = df[v]

    colnames = set(df.columns)
    if preprocessors:
        for k, v in preprocessors.items():
            if k not in fields:
                continue
            df[k] = df[k].apply(v)

    df = df[fields]
    if "orthography" in fields:
        df["length"] = [len(x) for x in df["orthography"]]
    if "frequency" in fields:
        df["frequency"] = pd.to_numeric(df["frequency"])

        f = list(set(fields.keys()) - {"frequency"})
        df = df.groupby(f, as_index=False).agg({"frequency": "sum"})

        f = df['frequency']
        min_freq = min(f[f > 0])
        f += min_freq
        df['log_frequency'] = np.log10(f)
        tot = f.sum()
        df['frequency_per_million'] = f * (1e6 / tot)
        df['zipf_score'] = np.log10(df['frequency_per_million'])

    return df.drop_duplicates().dropna(axis=0)
