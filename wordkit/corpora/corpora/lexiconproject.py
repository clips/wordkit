"""Readers for lexicon project corpora."""
import os
from ..base import reader
from ..base.utils import _calc_hash


AUTO_LANGUAGE = {"blp": "eng-uk",
                 "elp": "eng-us",
                 "flp": "fra",
                 "dlp1": "nld",
                 "dlp2": "nld",
                 "clp": "chi",
                 "klp": "kor"}

PROJECT_SEP = {"blp": "\t", "dlp2": "\t", "elp": ","}
PROJECT2FIELD = {"dlp1": {"orthography": "spelling"},
                 "dlp2": {"orthography": "spelling", "rt": "rtC.mean"},
                 "blp": {"orthography": "spelling"},
                 "elp": {"orthography": "Word", "rt": "I_Mean_RT"},
                 "flp": {"orthography": "item"},
                 "clp": {"orthography": "Character", "rt": "RT"},
                 "klp": {"orthography": "Stimuli",
                         "frequency": "Freq",
                         "lexicality": "Lexicality",
                         "rt": "Stim_RT_M"}}
AUTO_PROJECT = {"french lexicon project words.xls": "flp",
                "blp-items.txt": "blp",
                "dlp-items.txt": "dlp1",
                "dlp2_items.tsv": "dlp2",
                "elp-items.csv": "elp",
                "chinese lexicon project sze et al.csv": "clp",
                "klp_ld_item_ver.1.0.xlsx": "klp"}

HASHES = {'4556a72ec2a4907a69ea3fdd2343ddf7': 'klp',
          '7c23effdd68df5a29d0662390c6f4f02': 'elp',
          '7c446dbd6f9a9e30be713b30b8c896c8': 'clp',
          '93eda69ab4479fbdcce271f6d49fd953': 'blp',
          '96da0efa380cccfbbd65c7baa16acd62': 'flp',
          'ee807ec2b01e269ae24e948d47f37574': 'dlp2',
          'fd604022fa808096610b6a2d60680589': 'dlp1'}


def lexiconproject(path,
                   fields=("rt", "orthography"),
                   language=None,
                   project=None):
    if project is None:
        try:
            hash = _calc_hash(path)
            project = HASHES[hash]
        except KeyError:
            try:
                project = AUTO_PROJECT[os.path.split(path)[-1]]
            except KeyError:
                raise ValueError("Your project is not correct. Allowed "
                                 f"projects are {set(PROJECT2FIELD.keys())}")
    else:
        try:
            project = AUTO_PROJECT[os.path.split(path)[-1]]
        except KeyError:
            raise ValueError("Your project is not correct. Allowed "
                             f"projects are {set(PROJECT2FIELD.keys())}")
    if language is None:
        try:
            language = AUTO_LANGUAGE[project]
        except KeyError:
            raise ValueError("You passed None to language, but we failed "
                             "to determine the language automatically.")

    return reader(path,
                  fields,
                  PROJECT2FIELD[project],
                  language,
                  sep=PROJECT_SEP.get(project, None))
