"""Improve the CELEX frequency norms."""
import numpy as np

from wordkit.corpora import celex, subtlex


if __name__ == "__main__":

    # In this tutorial we merge the celex and subtlex corpora.
    # This allows us to update the out-dated frequencies from celex.
    c = celex("path_to_celex", fields=("orthography", "phonology"))
    s = subtlex("path_to_subtlex", fields=("orthography", "frequency"))

    # We concatenate both corpora
    merged = (c + s)

    # And then aggregate the merged corpus by orthography
    # The first argument is the set by which to aggregate.
    # This can be a tuple or a single field name.
    # The second argument specifies which columns are merged.
    # The third argument specifies which columns are kept.
    # The fourth argument specifies how to merge the merged columns.
    merged = merged.aggregate("orthography",
                              "frequency",
                              "phonology",
                              np.sum)

    # The first 10 items
    print(merged[:10])
