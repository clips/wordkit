"""Readers for lexicon project corpora."""
import os
from ..base import reader


ALLOWED_LANGUAGES = ("eng-uk", "eng-us", "fra", "nld", "chi", "kor")
LANG_SEP = {"eng-uk": "\t", "nld": "\t", "eng-us": ","}
language2field = {"nld": {"orthography": "spelling"},
                  "eng-uk": {"orthography": "spelling"},
                  "eng-us": {"orthography": "Word", "rt": "I_Mean_RT"},
                  "fra": {"orthography": "item"},
                  "chi": {"orthography": "Character", "rt": "RT"},
                  "kor": {"orthography": "Stimuli",
                          "frequency": "Freq",
                          "lexicality": "Lexicality",
                          "rt": "Stim_RT_M"}}
AUTO_LANGUAGE = {"french lexicon project words.xls": "fra",
                 "blp-items.txt": "eng-uk",
                 "dlp-items.txt": "nld",
                 "elp-items.csv": "eng-us",
                 "chinese lexicon project sze et al.csv": "chi",
                 "klp_ld_item_ver.1.0.xlsx": "kor"}


def lexiconproject(path, fields=("rt", "orthography"), language=None):
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
    if language not in language2field:
        langs = set(language2field.keys())
        raise ValueError("Your language {}, was not in the set of "
                         "allowed languages: {}".format(language,
                                                        langs))
    return reader(path,
                  fields,
                  language2field[language],
                  language,
                  sep=LANG_SEP.get(language, None))
