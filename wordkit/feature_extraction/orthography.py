"""Functions for extracting orthographical features."""
import numpy as np


VOWELS = set("aeioujy")
CONSONANTS = set("bcdfghklmnpqrstvwxz")
BOTH = CONSONANTS.union(VOWELS)


def extract_one_hot_characters(characters):
    """
    Extract one-hot encoded binary vectors for your characters.

    Each character will be assigned a unique one-hot encoded binary
    vector. Words transformed using this feature set will thus be
    matrices of one-hot encoded vectors.

    Parameters
    ----------
    characters : string or list
        The unique characters occurring in your dataset.

    Returns
    -------
    features : dict
        A dictionary mapping from characters to one-hot encoded arrays.

    """
    binary_features = np.eye(len(characters))
    return {l: binary_features[idx]
            for idx, l in enumerate(characters)}


def extract_consonant_vowel_characters(characters):
    """
    Split sets of characters into consonant and vowel character sets.

    This is useful for testing whether characters themselves are already
    categorized into some kind of Consonant Vowel set, or are already split
    into syllables.

    Note that this functions throws an error if a character is neither a
    consonant nor a vowel.

    Parameters
    ----------
    characters : string or list
        The unique characters occuring in your dataset.

    Returns
    -------
    vowels : list
        The list of characters which are vowels.
    consonants : list
        The list of characters which are consonants.

    """
    diff = set([x.lower() for x in characters]) - BOTH

    if diff:
        raise ValueError("{} was in your character set, but is not a vowel or"
                         " consonant.".format(diff))

    vowels = [x for x in characters if x.lower() in VOWELS]
    consonants = [x for x in characters if x.lower() in CONSONANTS]

    return vowels, consonants
