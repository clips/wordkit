import re

import numpy as np

diacritics = {
    "ː",
    "̤",
    "ˠ",
    "̠",
    "̈",
    "̞",
    "̩",
    "̻",
    "ʰ",
    "ʼ",
    "̝",
    "ʲ",
    "̥",
    "̟",
    "ˤ",
    "̃",
    "̺",
    "͡",
    "̯",
    "̪",
    "̰",
    "ʷ",
}


remove_double = re.compile(r"(ː)(\1){1,}")


def apply_if_not_na(x, func):
    """Applies function to something if it is not NA."""
    try:
        return x if np.isnan(x) else func(x)
    except TypeError:
        return func(x)


def segment_phonology(phonemes, items=diacritics, to_keep=diacritics):
    """
    Segment a list of characters into chunks by joining diacritics.

    This function turns a string of phonemes, which might have diacritics and
    suprasegmentals as separate characters, into a list of phonemes in which
    any diacritics or suprasegmentals have been joined.

    It also removes any diacritics which are not in the list to_keep,
    allowing the user to systematically remove any distinction which is not
    deemed useful for the current research.

    Additionally, any double longness markers are removed, and replaced by
    a single marker.

    Parameters
    ----------
    phonemes : list
        A list of phoneme characters to segment.

    items : list
        A list of characters which to treat as diacritics.

    to_keep : list
        A list of diacritics from the list passed to items which to keep.
        Any items in this list are not removed as spurious diacritics.
        If to_keep and items are the same list, all items are kept.

    """
    phonemes = remove_double.sub(r"\g<1>", phonemes)
    phonemes = [list(p) for p in phonemes]
    idx = 0
    while idx < len(phonemes):
        x = phonemes[idx]
        if x[0] in items:
            if x[0] in to_keep:
                phonemes[idx - 1].append(x[0])
            phonemes.pop(idx)
        else:
            idx += 1

    return tuple(["".join(x) for x in phonemes if x])
