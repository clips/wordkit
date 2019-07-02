"""A frame class."""
import numpy as np
import json

from copy import deepcopy
from collections import defaultdict
from itertools import chain
from .utils import (calc_length,
                    calc_fpm_score,
                    calc_zipf_score,
                    calc_log_frequency)


def not_nan_or_none(x):
    """Check whether a given input is either nan or None."""
    if np.ndim(x) > 2:
        return np.ones(x.shape[0], dtype=bool)
    x = np.asarray(x)
    m1 = x != None # noqa
    try:
        # Looks stupid, is correct
        m1[m1] &= ~np.isnan(x[m1].astype(float))
        return m1
    except (TypeError, ValueError):
        return m1


class Frame(list):
    """A Frame class."""

    prep = {"zipf_score": calc_zipf_score,
            "log_frequency": calc_log_frequency,
            "frequency_per_million": calc_fpm_score,
            "length": calc_length}

    def __add__(self, x):
        """Override add to return Frame."""
        return type(self)(super().__add__(x))

    def __eq__(self, x):
        """Check for equality."""
        if len(self) != len(x):
            return False
        for a, b in zip(self, x):
            if a != b:
                return False
        return True

    def _check_dict(self, x):
        for idx, item in enumerate(x):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx}: {x} was not a dictionary. A "
                                 "Frame requires all items to be "
                                 "dictionaries.")

    def __iadd__(self, other):
        self._check_dict(other)
        return super().__iadd__(other)

    def __getitem__(self, x):
        """Getter that returns a frame instead of a list."""
        if isinstance(x, str):
            return self.get(x)
        if isinstance(x, (np.ndarray, list)):
            result = []
            if all([isinstance(x_, str) for x_ in set(x)]):
                res = zip(*[self.get(x_) for x_ in x])
                result = [dict(zip(x, x_)) for x_ in res]
                return type(self)(result)
            if isinstance(x, list):
                x = np.asarray(x)
            if x.dtype == np.bool:
                if len(x) != len(self):
                    raise ValueError("Boolean indexing is only allowed when "
                                     "the length of the boolean array matches "
                                     f"the length of the Frame: got {len(x)}, "
                                     f"expected {len(self)}")
                x = np.flatnonzero(x)
            for idx in x:
                result.append(super().__getitem__(idx))
        else:
            result = super().__getitem__(x)
        # Only got a single result back
        if isinstance(result, dict):
            return result
        return type(self)(result)

    def __setitem__(self, x, item):
        """Setter."""
        if isinstance(x, int):
            self._check_dict([x])
            super().__setitem__(x, item)

        elif isinstance(x, str):
            if isinstance(item, (np.ndarray, list, tuple)):
                if len(item) != len(self):
                    raise ValueError(f"Your list of items to add was not "
                                     f"the same length as your WordStore: "
                                     f"got {len(item)}, expected {len(self)}.")
                mask = not_nan_or_none(item)
                for idx in np.flatnonzero(mask):
                    self[idx][x] = item[idx]
            else:
                raise ValueError(f"You indexed a column, but tried to assign "
                                 f"a {type(item)}. Expected ndarray or list")
        else:
            raise ValueError(f"You passed an illegal combination of things. "
                             f"x = {x} with type {type(x)}; item = {item} with"
                             f" type {type(item)} ")

    def append(self, x):
        """Append function with check."""
        self._check_dict([x])
        super().append(x)

    def extend(self, x):
        """Append function with check."""
        self._check_dict(x)
        super().extend(x)

    def get(self, key):
        """
        Gets values of a key from all words in the Frame.

        Parameters
        ----------
        key : object
            The key for which to retrieve all items.
        na_value : None or object
            The value to insert if a key is not present.

        Returns
        -------
        values : np.array
            The value of the key to retrieve.

        """
        if key not in self.columns and key in self.prep:
            self[key] = self.prep[key](self)
        X = np.array([x.get(key, None) for x in self])
        try:
            X = X.astype(float)
        except ValueError:
            pass
        return X

    def where(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            This function also takes general keyword arguments that take
            keys as keys and functions as values.

            e.g. if "frequency" is a field, you can use
                frequency=lambda x: x > 10
            as a keyword argument to only retrieve items with a frequency > 10.

        Returns
        -------
        f : Frame

        """
        # Only if we actually have functions should we do something
        items = self
        iters = {k: set(v) for k, v in kwargs.items()
                 if isinstance(v, (tuple, set, list))}
        callables = {k: v for k, v in kwargs.items()
                     if callable(v)}
        singles = {k: v for k, v in kwargs.items()
                   if k not in iters and k not in callables}
        if singles:
            iter_singles = iter(singles.items())
            k, v = next(iter_singles)
            conjunct = items[k] == v
            for k, v in iter_singles:
                conjunct &= items[k] == v
            items = [i for c, i in zip(conjunct, items) if c]

        for k, v in iters.items():
            items = [i for i in items if i[k] in v]
        for k, v in callables.items():
            items = [i for i in items if v(i[k])]

        return type(self)(items)

    @property
    def columns(self):
        """List all columns."""
        c = set()
        for x in self:
            c.update(x)

        return c

    def filter_nan(self, columns):
        """Simple nan filtering."""
        if isinstance(columns, str):
            columns = (columns,)
        not_nan = np.ones(len(self), dtype=np.bool)
        for x in columns:
            not_nan &= not_nan_or_none(self[x])
        return type(self)(self[not_nan])

    def aggregate(self, columns, columns_to_merge, func=np.sum):
        """Collapses the Frame by adding duplicates together."""
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns_to_merge, str):
            columns_to_merge = [columns_to_merge]
        if columns is None:
            columns = set(self.columns) - set(columns_to_merge)

        keys = [self.get(x) for x in columns]
        not_none = np.ones(len(keys[0]), dtype=bool)
        for k in keys:
            not_none &= k != None # noqa
        real_idx = np.flatnonzero(not_none)
        keys = np.stack(keys)[:, not_none].astype(str)

        _, idxes, inverse, c = np.unique(keys,
                                         axis=1,
                                         return_inverse=True,
                                         return_index=True,
                                         return_counts=True)
        # All unique items, sorted
        # Inverse is created with reference to this order.
        idxes = real_idx[idxes]
        joined = list(chain(columns, columns_to_merge))
        new = self[idxes][joined].copy()

        for idx, f in enumerate(columns_to_merge):
            vals = self.get(f)[not_none]
            mask = not_nan_or_none(vals)

            # Also mask the inverse array
            loc_inv = inverse[mask]
            res = np.zeros(len(new), dtype=object)
            res_ = defaultdict(list)
            not_touched = np.ones(len(idxes), dtype=bool)

            for idx, inv_idx in zip(np.arange(len(vals))[mask], loc_inv):
                not_touched[inv_idx] = False
                res_[inv_idx].append(vals[idx].tolist())
            for k, v in res_.items():
                res[k] = func(v)

            res[not_touched] = None
            # TODO: check return types
            new[f] = res

        return new

    def save(self, path):
        """Save to file as a JSON file."""
        json.dump(self, open(path, 'w'))

    def copy(self):
        """Return a copy."""
        return deepcopy(self)

    def drop(self, keys):
        """Drops one or more columns."""
        for x in self:
            for k in keys:
                try:
                    x.pop(k)
                except KeyError:
                    pass
        return self

    def transform(self, key, function):
        """Transforms."""
        for x in self:
            x[key] = function(x[key])
        return self
