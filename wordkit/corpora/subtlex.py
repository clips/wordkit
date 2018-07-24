"""Corpus readers for Subtlex."""
from .base import Reader
import pandas as pd
import os

field_ids = {"orthography": 0, "frequency": 1, "log_frequency": None}
# Currently redundant, but useful for future-proofing.
language2field = {"eng-uk": field_ids,
                  "eng-us": field_ids,
                  "nld": field_ids,
                  "esp": field_ids,
                  "deu": field_ids,
                  "chi": field_ids}


MAX_FREQ = {"eng-uk": 201863977,
            "eng-us": 49766042,
            "nld": 43343958,
            "esp": 9264447,
            "deu": 21126690,
            "chi": 33645637}


MAX_FREQ = {k: v / 1000000 for k, v in MAX_FREQ.items()}
ALLOWED_LANGUAGES = set(MAX_FREQ.keys())


class Subtlex(Reader):
    """
    Reader for the various Subtlex corpora.

    These are corpora of frequency norms which explain significantly more
    variance than other frequency norms based on large corpora.

    In general, all the Subtlex corpora are associated with a paper.
    For an overview, visit:
        http://crr.ugent.be/programs-data/subtitle-frequencies

    Please make sure to read the associated articles, and cite them whenever
    you use a version of Subtlex!

    This class can read both the Excel and csv versions of the subtlex files.
    If you can, please consider converting the Excel files offered by the
    original authors to a csv before processing as this will significantly
    speed up the entire process.

    Parameters
    ----------
    path : string
        The path to the file. The extension of this file will be used to
        determine the method we use to open the file.

    fields : tuple, default ("orthography", "frequency")
        The fields to retrieve from this corpus.

    language : string, default "eng-uk"
        The language the corpus is in.

    """

    def __init__(self,
                 path,
                 fields=("orthography", "frequency", "log_frequency"),
                 language="eng-uk"):
        """Initialize the subtlex reader."""
        if language not in ALLOWED_LANGUAGES:
            raise ValueError("Your language {}, was not in the set of "
                             "allowed languages: {}".format(language,
                                                            ALLOWED_LANGUAGES))
        super().__init__(path,
                         fields,
                         language2field[language],
                         language,
                         merge_duplicates=True,
                         diacritics=None,
                         frequency_divider=MAX_FREQ[language])

    def _open(self):
        """Open the file for reading."""
        if os.path.splitext(self.path)[1].startswith(".xls"):
            f = pd.read_excel(self.path)
        else:
            f = pd.read_csv(self.path, sep="\t")

        data = f.as_matrix().tolist()
        if self.language == "chi":
            data = data[2:]

        return data

    def _retrieve(self, iterable, wordlist=None, **kwargs):
        """Retrieve the word-frequency pairs from the Subtlex database."""
        wordlist = {x.lower() for x in wordlist}

        use_log = "log_frequency" in self.fields
        use_freq = "frequency" in self.fields
        use_orth = "orthography" in self.fields

        for line in iterable:

            word = {}

            orth = line[self.field_ids['orthography']]
            if isinstance(orth, float):
                continue
            if wordlist and orth not in wordlist:
                continue
            if use_orth:
                word["orthography"] = orth

            if use_freq or use_log:
                freq = line[self.field_ids['frequency']]
                word["frequency"] = freq

            yield word
