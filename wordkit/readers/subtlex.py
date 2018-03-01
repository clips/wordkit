"""Corpus readers for Subtlex."""
from .base import Reader, identity
import pandas as pd
import os


field_ids = {"orthography": 0, "frequency": 1}

MAX_FREQ = {"eng-uk": 201863977,
            "eng-us": 49705658,
            "nld": 43209236,
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
    language : string, default "english"
        The language the corpus is in.
    filter_function : function, default identity
        The function used to filter the output.

    """

    def __init__(self,
                 path,
                 language,
                 fields=("orthography", "frequency"),
                 filter_function=identity):
        """Initialize the subtlex reader."""
        if language not in ALLOWED_LANGUAGES:
            raise ValueError("Your language {}, was not in the set of "
                             "allowed languages: {}".format(language,
                                                            ALLOWED_LANGUAGES))
        super().__init__(path,
                         fields,
                         field_ids,
                         language,
                         merge_duplicates=True,
                         filter_function=filter_function,
                         diacritics=None)

    def _retrieve(self, wordlist=None, **kwargs):
        """Retrieve the word-frequency pairs from the Subtlex database."""
        result = []
        wordlist = {x.lower() for x in wordlist}

        if os.path.splitext(self.path)[1].startswith(".xls"):
            f = pd.read_excel(self.path)
        else:
            f = pd.read_csv(self.path, sep="\t")

        data = f.as_matrix().tolist()
        if self.language == "chi":
            data = data[2:]

        for line in data:

            orth = line[self.fields['orthography']]
            if isinstance(orth, float):
                continue
            if wordlist and orth not in wordlist:
                continue

            freq = line[self.fields['frequency']] / MAX_FREQ[self.language]
            result.append({"orthography": orth, "frequency": freq})

        return result
