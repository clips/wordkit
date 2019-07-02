"""Corpus readers for Subtlex."""
import os
from ..base import reader

# Currently redundant, but useful for future-proofing.
language2field = {"eng-uk": {"orthography": "Spelling",
                             "frequency": "FreqCount"},
                  "eng-us": {"orthography": "Word",
                             "frequency": "FREQcount"},
                  "nld": {"orthography": "Word",
                          "frequency": "FREQcount"},
                  "deu": {"orthography": "Word",
                          "frequency": "WFfreqCount"},
                  "chi": {"orthography": "Word",
                          "frequency": "WCount"}}

ALLOWED_LANGUAGES = set(language2field.keys())
AUTO_LANGUAGE = {'subtlex-ch-wf.xlsx': 'chi',
                 'subtlex-de cleaned with google00 frequencies.xlsx': 'deu',
                 'subtlex-nl.cd-above2.txt': 'nld',
                 'subtlex-uk.xlsx': 'eng-uk',
                 'subtlexusfrequencyabove1.xls': 'eng-us'}


def subtlex(path,
            fields=("orthography", "frequency"),
            language=None):
    """Initialize the subtlex reader."""
    if language is None:
        try:
            language = AUTO_LANGUAGE[os.path.split(path)[1].lower()]
        except KeyError:
            raise ValueError("You passed None to language, but we failed "
                             "to determine the language automatically.")
    else:
        try:
            if AUTO_LANGUAGE[os.path.split(path)[1]] != language:
                raise ValueError("Your language is {}, but your filename "
                                 "belongs to another language."
                                 "".format(language))
        except KeyError:
            pass
    if language not in ALLOWED_LANGUAGES:
        raise ValueError("Your language {}, was not in the set of "
                         "allowed languages: {}".format(language,
                                                        ALLOWED_LANGUAGES))
    if language == "chi":
        skiprows = 2
    else:
        skiprows = 0

    return reader(path,
                  fields,
                  language2field[language],
                  language,
                  sep="\t",
                  skiprows=skiprows)
