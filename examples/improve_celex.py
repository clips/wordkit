"""Improve the CELEX frequency norms."""
import pandas as pd

from wordkit.corpora import celex_english, subtlexuk


if __name__ == "__main__":

    # In this tutorial we merge the celex and subtlex corpora.
    # This allows us to update the out-dated frequencies from celex.
    c = celex_english("path_to_celex", fields=("orthography", "phonology"))
    s = subtlexuk("path_to_subtlex", fields=("orthography", "frequency"))

    # Merge the corpora using pandas
    merged = pd.merge(c, s)

    # The first 10 items
    print(merged[:10])
