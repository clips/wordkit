"""Base class for corpus readers."""
import os
import pandas as pd

from collections import defaultdict
from itertools import chain
from .frame import Frame


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
            print(kwargs)
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

    return df.to_dict("records")


def reader(path,
           fields,
           field_ids,
           language,
           preprocessors=None,
           opener=_open,
           **kwargs):
    """Init the base class."""
    if not os.path.exists(path):
        raise FileNotFoundError("The file you specified does not "
                                f"exist: {path}")
    if isinstance(fields, str):
        fields = (fields,)

    df = Frame(opener(path, **kwargs))
    # Columns in dataset
    colnames = set(df.columns)
    if fields:
        rev = defaultdict(list)
        for k, v in field_ids.items():
            rev[v].append(k)
        c = set(chain.from_iterable([rev.get(x, [x]) for x in colnames]))
        redundant = set(fields) - c
        if redundant:
            raise ValueError("You passed fields which were not in "
                             f"the dataset {redundant}. The available fields "
                             f"are: {c}")
    fields = {k: field_ids.get(k, k) for k in fields}

    for k, v in ((k, v) for k, v in fields.items() if k != v):
        df[k] = df.get(v)

    colnames = set(df.columns)
    other_fields = colnames - set(fields)
    df.drop(other_fields)
    if preprocessors:
        for k, v in preprocessors.items():
            if k not in fields:
                continue
            df.transform(k, v)

    return df
