"""Functions for extracting and handling phonological features."""
import numpy as np
from itertools import chain
from functools import partial
from collections import defaultdict

from ipapy.ipastring import IPAString

DEFAULT = "ɡæsɪrbʌɒɑtyðəvepʒuhʃoxdɛfiwθjlɔʊmnaŋɜkz"


def one_hot_phonemes(phonemes=DEFAULT,
                     use_long=False):
    """
    Encode phonemes as one-hot vectors.

    Given a string of unicode phonemes, this function divides them up
    into consonants and vowels, and then assigns each unique phoneme a
    one-hot binary vector.

    Parameters
    ==========
    phonemes : list, optional, default
        The phonemes used.

    Returns
    =======
    features : tuple
        A tuple of dictionaries. The first dictionary contains the vowels and
        their features, the second dictionary the consonants and their
        features.

    Examples
    ========
    # For default features
    >>> features = one_hot_phonemes()

    """
    phonemes = IPAString(unicode_string="".join(phonemes),
                         single_char_parsing=True)
    phonemes = [p for p in phonemes if not p.is_diacritic]

    vowels = [p.unicode_repr for p in phonemes if p.is_vowel]
    consonants = [p.unicode_repr for p in phonemes if not p.is_vowel]

    num_vowels = len(vowels)
    if use_long:
        num_vowels *= 2

    vowel_dict = defaultdict(partial(np.zeros, num_vowels))

    for idx, p in enumerate(vowels):

        vowel_dict[p][idx] = 1.
        if use_long:
            vowel_dict[p+'ː'][idx + len(vowels)] = 1.

    num_consonants = len(consonants)
    if use_long:
        num_consonants *= 2

    consonant_dict = defaultdict(partial(np.zeros, num_consonants))

    for idx, p in enumerate(consonants):

        consonant_dict[p][idx] = 1.
        if use_long:
            consonant_dict[p+'ː'][idx + len(consonants)] = 1.

    return vowel_dict, consonant_dict


def extract_phoneme_features(phonemes=DEFAULT,
                             use_is_vowel=True,
                             use_place=True,
                             use_manner=True,
                             use_voicing=True,
                             use_backness=True,
                             use_height=True,
                             use_longness=True):
    """
    Extract symbolic features from your phonemes.

    This function associates the feature strings associated with each
    phoneme in your dataset.

    Parameters
    ----------
    phonemes : list, optional, default
        A string or other iterable of IPA characters you want to use.
    use_is_vowel: bool, optional, default True
        Whether to extract vowelness as a feature.
    use_place: bool, optional, default True
        Whether to use place as a feature.
    use_manner: bool, optional, default True
        Whether to use manner as a feature.
    use_voicing: bool, optional, default True
        Whether to use voicing as a feature.
    use_backness: bool, optional, default True
        Whether to use backness as a feature.
    use_height: bool, optional, default True
        Whether to use height as a feature.

    Returns
    -------
    features: tuple
        A tuple of dictionaries. The first dictionary contains the vowels and
        their features, the second dictionary the consonants and their
        features.

    Example
    -------
    >>> features = extract_phoneme_features()

    """
    phonemes = IPAString(unicode_string="".join(phonemes),
                         single_char_parsing=True)

    phonemes = [p for p in phonemes
                if not p.is_diacritic or p.is_suprasegmental]

    vowels = filter(lambda x: x.is_vowel, phonemes)
    consonants = filter(lambda x: not x.is_vowel, phonemes)

    vowels = {p.unicode_repr: extract_single_phoneme(p,
                                                     use_is_vowel,
                                                     use_place,
                                                     use_manner,
                                                     use_voicing,
                                                     use_backness,
                                                     use_height)
              for p in vowels}

    consonants = {p.unicode_repr: extract_single_phoneme(p,
                                                         use_is_vowel,
                                                         use_place,
                                                         use_manner,
                                                         use_voicing,
                                                         use_backness,
                                                         use_height)
                  for p in consonants}

    if use_longness:
        vowels = {k: v + ['short'] for k, v in vowels.items()}
        long_vowels = {"{}ː".format(k): v[:-1] + ['long']
                       for k, v in vowels.items()}
        vowels.update(long_vowels)

        consonants = {k: v + ['short'] for k, v in consonants.items()}
        long_consonants = {"{}ː".format(k): v[:-1] + ['long']
                           for k, v in consonants.items()}
        consonants.update(long_consonants)

    return vowels, consonants


def extract_single_phoneme(phoneme,
                           use_is_vowel,
                           use_place,
                           use_voicing,
                           use_backness,
                           use_manner,
                           use_height):
    """
    Extract symbolic features from a single phoneme.

    Parameters
    ----------
    phoneme : IPAChar
        A single IPA character from which to extract features
    use_is_vowel: bool, optional, default True
        Whether to extract vowelness as a feature.
    use_place: bool, optional, default True
        Whether to use place as a feature.
    use_manner: bool, optional, default True
        Whether to use manner as a feature.
    use_voicing: bool, optional, default True
        Whether to use voicing as a feature.
    use_backness: bool, optional, default True
        Whether to use backness as a feature.
    use_height: bool, optional, default True
        Whether to use height as a feature.

    Returns
    -------
    features: list
        A list of strings, representing the symbolic features associated
        with this phoneme according to the IPA specification.

    Example
    -------
    >>> from ipapy import IPA_CHARS
    >>> features = extract_single_phoneme(IPA_CHARS[0])

    """
    features = list()

    if use_is_vowel:
        if phoneme.is_vowel:
            features.append("vowel")
        else:
            features.append("consonant")
    if use_place:
        if not phoneme.is_vowel:
            features.append(phoneme.place)
    if use_voicing:
        if not phoneme.is_vowel:
            features.append(phoneme.voicing)
    if use_backness:
        if phoneme.is_vowel:
            features.append(phoneme.backness)
    if use_manner:
        if not phoneme.is_vowel:
            features.append(phoneme.manner)
    if use_height:
        if phoneme.is_vowel:
            features.append(phoneme.height)

    return features


def phoneme_features(features,
                     phonemes=list(DEFAULT),
                     use_is_vowel=True,
                     use_place=True,
                     use_manner=True,
                     use_voicing=True,
                     use_backness=True,
                     use_height=True,
                     use_longness=True):
    """
    Replace symbolic features by a set of predefined feature vectors.

    Parameters
    ----------
    phonemes : list
        A single IPA character from which to extract features
    use_is_vowel: bool, optional, default True
        Whether to extract vowelness as a feature.
    use_place: bool, optional, default True
        Whether to use place as a feature.
    use_manner: bool, optional, default True
        Whether to use manner as a feature.
    use_voicing: bool, optional, default True
        Whether to use voicing as a feature.
    use_backness: bool, optional, default True
        Whether to use backness as a feature.
    use_height: bool, optional, default True
        Whether to use height as a feature.

    Returns
    -------
    features: tuple
        A tuple of dictionaries. The first dictionary contains the vowels and
        their features, the second dictionary the consonants and their
        features.

    """
    # Not all feature sets explicitly code vowelness or consonantness
    # but we don't want to crash the system if the user desires these features.
    if 'vowel' not in features:
        features['vowel'] = (1,)
    if 'consonant' not in features:
        features['consonant'] = (0,)

    if 'long' not in features:
        features['long'] = (1,)

    if 'short' not in features:
        features['short'] = (0,)

    vowels, consonants = extract_phoneme_features(phonemes,
                                                  use_is_vowel,
                                                  use_place,
                                                  use_manner,
                                                  use_voicing,
                                                  use_backness,
                                                  use_height,
                                                  use_longness)

    vowels = {k: np.concatenate([features[x] for x in v])
              for k, v in vowels.items()}

    consonants = {k: np.concatenate([features[x] for x in v])
                  for k, v in consonants.items()}

    return vowels, consonants


def one_hot_phoneme_features(phonemes=DEFAULT,
                             use_is_vowel=True,
                             use_place=True,
                             use_manner=True,
                             use_voicing=True,
                             use_backness=True,
                             use_height=True):
    """
    Replace symbolic features by one-hot encoded features.

    If, for example, the dimension "backness" is selected, and this dimension
    happens to have five different configurations, we will reserve five
    binary variables for backness.

    As such, this featurization technique gives very large feature spaces,
    but are ideal if the user doesn't want to assume any linguistic theory.

    Parameters
    ----------
    phoneme : IPAChar
        A single IPA character from which to extract features
    use_is_vowel: bool, optional, default True
        Whether to extract vowelness as a feature.
    use_place: bool, optional, default True
        Whether to use place as a feature.
    use_manner: bool, optional, default True
        Whether to use manner as a feature.
    use_voicing: bool, optional, default True
        Whether to use voicing as a feature.
    use_backness: bool, optional, default True
        Whether to use backness as a feature.
    use_height: bool, optional, default True
        Whether to use height as a feature.

    Returns
    -------
    features: tuple
        A tuple of dictionaries. The first dictionary contains the vowels and
        their features, the second dictionary the consonants and their
        features.

    """
    vowels, consonants = extract_phoneme_features(phonemes,
                                                  use_is_vowel,
                                                  use_place,
                                                  use_manner,
                                                  use_voicing,
                                                  use_backness,
                                                  use_height)

    vowel_features = set(chain.from_iterable(vowels.values()))
    consonant_features = set(chain.from_iterable(consonants.values()))

    vowel_feature_space = np.eye(len(vowel_features))
    consonant_feature_space = np.eye(len(consonant_features))

    vowel_features = {k: vowel_feature_space[idx]
                      for idx, k in enumerate(vowel_features)}

    consonant_features = {k: consonant_feature_space[idx]
                          for idx, k in enumerate(consonant_features)}

    vowel_features.update(consonant_features)

    return phoneme_features(vowel_features,
                            phonemes,
                            use_is_vowel,
                            use_place,
                            use_manner,
                            use_voicing,
                            use_backness,
                            use_height)
