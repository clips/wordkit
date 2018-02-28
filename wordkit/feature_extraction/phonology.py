"""Functions for extracting and handling phonological features."""
import numpy as np
from functools import reduce
from sklearn.feature_extraction import DictVectorizer

from ipapy.ipastring import IPAString

DEFAULT = tuple("ɡæsɪrbʌɒɑtyðəvepʒuhʃoxdɛfiwθjlɔʊmnaŋɜkz")
FORBIDDEN_DESCRIPTORS = {"suprasegmental", "vowel", "consonant", "diacritic"}


def phoneme_set_to_string(phonemes):
    """Convert a phoneme set (a tuple) into a single IPA character."""
    return ["".join([x.unicode_repr for x in p])
            for p in phonemes]


def parse_phonemes(phonemes):
    """Parse the incoming tuple of phonemes as IPA characters."""
    phonemes = [IPAString(unicode_string=p) for p in phonemes]

    vowels = filter(lambda x: x[0].is_vowel, phonemes)
    consonants = filter(lambda x: not x[0].is_vowel, phonemes)

    return list(vowels), list(consonants)


def phoneme_descriptors(phonemes, forbidden=FORBIDDEN_DESCRIPTORS):
    """Retrieve the set of descriptors for complex phonemes."""
    descriptors = []

    for p in phonemes:
        desc = reduce(set.union, [p.descriptors for p in p], set())
        desc -= forbidden
        descriptors.append(desc)

    return descriptors, reduce(set.union, descriptors, set())


def grouped_phoneme_descriptors(phonemes, allowed=None):
    """Retrieve the set of descriptors for complex phonemes."""
    diacritic_descriptors = set()
    results = []
    for p in phonemes:
        result = {}
        if p[0].is_consonant:
            result["manner"] = p[0].manner
            result["place"] = p[0].place
            result["voicing"] = p[0].voicing
        else:
            result["backness"] = p[0].backness
            result["height"] = p[0].height
            result["roundness"] = p[0].roundness
        for diacritic in p[1:]:
            desc = set(diacritic.descriptors) - FORBIDDEN_DESCRIPTORS
            diacritic_descriptors.update(desc)
            for x in desc:
                result[x] = x

        if allowed is not None:
            result = {k: v for k, v in result.items() if v in allowed}

        results.append(result)

    for x in results:
        for key in diacritic_descriptors - x.keys():
            x[key] = "absent"

    return results


def phoneme_feature_vectors(phonemes, forbidden=FORBIDDEN_DESCRIPTORS):
    """
    Create feature vectors for phonemes on the basis of their descriptors.

    A descriptor is a string description of a quality a certain phoneme
    possesses. Examples of descriptions are "back", "central", "near-close".

    This function first gathers all descriptors for each phoneme, and then
    creates a single binary feature for each unique descriptor.
    Each phoneme is then assigned a 1 for features for which it possesses the
    descriptor, and 0 for features for which it doesn't possess the
    descriptor.

    Parameters
    ----------
    phonemes : tuple
        A tuple of strings, where each string is a phoneme. Phonemes can
        consist of multiple characters.
    forbidden : set
        descriptors from this set are filtered out. The standard set of
        forbidden descriptors contains "suprasegmental", "vowel", and
        "consonant".

    """
    descriptors, all_descriptors = phoneme_descriptors(phonemes,
                                                       FORBIDDEN_DESCRIPTORS)

    all_descriptors = {v: idx for idx, v in enumerate(all_descriptors)}

    phoneme = np.zeros((len(descriptors),
                        len(all_descriptors)))

    for idx, p in enumerate(descriptors):
        indices = [all_descriptors[d] for d in p]
        phoneme[idx, indices] = 1

    return phoneme


def one_hot_phonemes(phonemes=DEFAULT):
    """
    Encode phonemes as one-hot vectors.

    Given a string of unicode phonemes, this function divides them up
    into consonants and vowels, and then assigns each unique phoneme a
    one-hot binary vector.

    Parameters
    ----------
    phonemes : list, optional, default
        The phonemes used.

    Returns
    -------
    features : tuple
        A tuple of dictionaries. The first dictionary contains the vowels and
        their features, the second dictionary the consonants and their
        features.

    Example
    -------
    # For default features
    >>> features = one_hot_phonemes()

    """
    vowels, consonants = parse_phonemes(phonemes)

    vowels = phoneme_set_to_string(vowels)
    consonants = phoneme_set_to_string(consonants)

    vowel_dict = dict(zip(vowels, np.eye(len(vowels))))
    consonant_dict = dict(zip(consonants, np.eye(len(consonants))))

    return vowel_dict, consonant_dict


def extract_phoneme_features(phonemes=DEFAULT):
    """
    Extract symbolic features from your phonemes.

    This function associates the feature strings associated with each
    phoneme in your dataset.

    Parameters
    ----------
    phonemes : list, optional, default
        A string or other iterable of IPA characters you want to use.

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
    vowels, consonants = parse_phonemes(phonemes)

    vowel_strings = phoneme_set_to_string(vowels)
    consonant_strings = phoneme_set_to_string(consonants)

    vowel_features = phoneme_feature_vectors(vowels)
    consonant_features = phoneme_feature_vectors(consonants)

    vowels = dict(zip(vowel_strings, vowel_features))
    consonants = dict(zip(consonant_strings, consonant_features))

    return vowels, consonants


def extract_grouped_phoneme_features(phonemes):
    """
    Extract phoneme features which are grouped per feature.

    This leads to a the same encoding as the extract_phoneme_features
    function.
    """
    vowels, consonants = parse_phonemes(phonemes)

    vowel_strings = phoneme_set_to_string(vowels)
    consonant_strings = phoneme_set_to_string(consonants)

    vowel_features = grouped_phoneme_descriptors(vowels)
    consonant_features = grouped_phoneme_descriptors(consonants)

    d = DictVectorizer(sparse=False)
    vowel_features = d.fit_transform(vowel_features)
    d = DictVectorizer(sparse=False)
    consonant_features = d.fit_transform(consonant_features)

    vowels = dict(zip(vowel_strings, vowel_features))
    consonants = dict(zip(consonant_strings, consonant_features))

    return vowels, consonants


def predefined_features(phonemes,
                        phoneme_features):
    """Use phonemes with pre-defined features."""
    vowels, consonants = parse_phonemes(phonemes)

    v_string = phoneme_set_to_string(vowels)
    c_string = phoneme_set_to_string(consonants)

    v_descriptors = grouped_phoneme_descriptors(vowels,
                                                phoneme_features.keys())
    c_descriptors = grouped_phoneme_descriptors(consonants,
                                                phoneme_features.keys())

    vowel_vectors = []

    for descriptors in v_descriptors:
        vec = []
        for k, v in sorted(descriptors.items()):
            if v != 'absent':
                vec.extend(phoneme_features[v])
            else:
                vec.append(0)
        vowel_vectors.append(vec)

    consonant_vectors = []

    for descriptors in c_descriptors:
        vec = []
        for k, v in sorted(descriptors.items()):
            if v != 'absent':
                vec.extend(phoneme_features[v])
            else:
                vec.append(0)
        consonant_vectors.append(vec)

    vowel_dict = dict(zip(v_string, vowel_vectors))
    consonant_dict = dict(zip(c_string, consonant_vectors))

    return vowel_dict, consonant_dict
