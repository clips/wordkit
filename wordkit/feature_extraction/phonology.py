"""Functions for extracting and handling phonological features."""
import numpy as np

from .base import BaseExtractor
from ipapy.ipastring import IPAString
from functools import reduce
from collections import defaultdict


FORBIDDEN_DESCRIPTORS = {"suprasegmental", "vowel", "consonant", "diacritic"}


class BasePhonemeExtractor(BaseExtractor):
    """Base class for the phoneme extractors."""

    @staticmethod
    def _phoneme_set_to_string(phonemes):
        """Convert a phoneme set (a tuple) into a single IPA character."""
        return ["".join([x.unicode_repr for x in p])
                for p in phonemes]

    @staticmethod
    def _parse_phonemes(phonemes):
        """Parse the incoming tuple of phonemes as IPA characters."""
        phonemes = [IPAString(unicode_string=p) for p in phonemes]

        vowels = filter(lambda x: x[0].is_vowel, phonemes)
        consonants = filter(lambda x: not x[0].is_vowel, phonemes)

        return list(vowels), list(consonants)

    @staticmethod
    def _phoneme_descriptors(phonemes, forbidden=FORBIDDEN_DESCRIPTORS):
        """Retrieve the set of descriptors for complex phonemes."""
        descriptors = []

        for p in phonemes:
            desc = reduce(set.union, [p.descriptors for p in p], set())
            desc -= forbidden
            descriptors.append(desc)

        return descriptors, reduce(set.union, descriptors, set())

    @staticmethod
    def _grouped_phoneme_descriptors(phonemes):
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

            results.append(result)

        for x in results:
            for key in diacritic_descriptors - x.keys():
                x[key] = "absent"

        return results

    def _phoneme_feature_vectors(self,
                                 phonemes,
                                 forbidden=FORBIDDEN_DESCRIPTORS):
        """
        Create feature vectors for phonemes on the basis of their descriptors.

        A descriptor is a string description of a quality a certain phoneme
        possesses. Examples of descriptions are "back", "central".

        This function first gathers all descriptors for each phoneme, and then
        creates a single binary feature for each unique descriptor.
        Each phoneme is then assigned a 1 for features for which it possesses
        the descriptor, and 0 for features for which it doesn't possess the
        descriptor.

        Parameters
        ----------
        phonemes : tuple
            A tuple of strings, where each string is a phoneme. Phonemes can
            consist of multiple characters.

        forbidden : set
            descriptors from this set are filtered out. The standard set of
            forbidden descriptors contains "suprasegmental", "vowel",
            "consonant", and "diacritic".

        """
        d, all_d = self._phoneme_descriptors(phonemes, FORBIDDEN_DESCRIPTORS)

        all_descriptors = {v: idx for idx, v in enumerate(all_d)}

        phoneme = np.zeros((len(d),
                            len(all_descriptors)))

        for idx, p in enumerate(d):
            indices = [all_descriptors[d] for d in p]
            phoneme[idx, indices] = 1

        return phoneme


class OneHotPhonemeExtractor(BasePhonemeExtractor):
    """
    An extractor which assigns each phoneme an orthogonal vector.

    Example
    -------
    >>> from wordkit.feature_extraction import OneHotPhonemeExtractor
    >>> phonemes = [('ə', 'b', 'e', 'ɪ', 's'),
                    ('k', 'ɔ', 'ŋ', 'ɡ', 'r', 'ɛ', 's')]
    >>> p = OneHotPhonemeExtractor()
    >>> vowels, consonants = p.extract(phonemes)

    """

    def __init__(self, field=None, include_space=True):
        """Initialize the extractor."""
        super().__init__(field)
        self.include_space = include_space

    def _process(self, phonemes):
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
            A tuple of dictionaries. The first dictionary contains the vowels
            and their features, the second dictionary the consonants and their
            features.

        """
        vowels, consonants = self._parse_phonemes(phonemes)

        vowels = self._phoneme_set_to_string(vowels)
        consonants = self._phoneme_set_to_string(consonants)

        if self.include_space:
            vowels.append(" ")
            consonants.append(" ")

        vowel_dict = dict(zip(vowels, np.eye(len(vowels))))
        consonant_dict = dict(zip(consonants, np.eye(len(consonants))))

        return vowel_dict, consonant_dict


class PhonemeFeatureExtractor(BasePhonemeExtractor):
    """
    An extractor which assigns each feature a binary value.

    This extractor uses the features defined in the IPA, and assigns each of
    them a dimension in a binary vector. Note that this means that internal
    coherence of features is not directly modeled, as every feature gets
    assigned the same importance.

    If you do desire to have internal coherence between feature groups, you
    can use the PredefinedFeatureExtractor with a group of predefined features.

    Example
    -------
    >>> from wordkit.feature_extraction import PhonemeFeatureExtractor
    >>> phonemes = [('ə', 'b', 'e', 'ɪ', 's'),
                    ('k', 'ɔ', 'ŋ', 'ɡ', 'r', 'ɛ', 's')]
    >>> p = PhonemeFeatureExtractor()
    >>> features = p.extract(phonemes)

    """

    def _process(self, phonemes):
        """
        Extract symbolic features from your phonemes.

        This function associates the feature strings associated with each
        phoneme in your dataset with a set of features.

        Parameters
        ----------
        phonemes : list, optional, default
            A string or other iterable of IPA characters you want to use.

        Returns
        -------
        features: tuple
            A tuple of dictionaries. The first dictionary contains the vowels
            and their features, the second dictionary the consonants and their
            features.

        """
        vowels, consonants = self._parse_phonemes(phonemes)

        vowel_strings = self._phoneme_set_to_string(vowels)
        consonant_strings = self._phoneme_set_to_string(consonants)

        vowel_features = self._phoneme_feature_vectors(vowels)
        consonant_features = self._phoneme_feature_vectors(consonants)

        vowels = dict(zip(vowel_strings, vowel_features))
        consonants = dict(zip(consonant_strings, consonant_features))

        return vowels, consonants


class PredefinedFeatureExtractor(BasePhonemeExtractor):
    """
    An extractor which used predefined features.

    The predefined features are a dictionary mapping features to associated
    feature vectors. Features, in this case, are names of distinctive features,
    not their associated groups.

    Parameters
    ----------
    phoneme_features : dict
        A dictionary from names of distinctive features, (e.g. 'plosive'),
        and the feature vector for these features. Note that these feature
        vectors must have the same length for each mutually exclusive category,
        but need not all be the same length.

        See ..features for examples

    field : str, default None
        The field to operate on.

    Example
    -------
    >>> from wordkit.feature_extraction import PredefinedFeatureExtractor
    >>> from wordkit.features import dislex_features
    >>> phonemes = [('ə', 'b', 'e', 'ɪ', 's'),
                    ('k', 'ɔ', 'ŋ', 'ɡ', 'r', 'ɛ', 's')]
    >>> p = PredefinedFeatureExtractor(dislext_features)
    >>> vowels, consonants = p.extract(phonemes)
    >>> # find out which groups were excluded, if any.
    >>> p.groups_excluded

    """

    def __init__(self,
                 phoneme_features,
                 field=None):
        """Initialize the extractor."""
        super().__init__(field=field)
        self.phoneme_features = phoneme_features
        self.feature_names = set(self.phoneme_features.keys())
        self.groups_excluded = []

    def _process(self, phonemes):
        """Use phonemes with pre-defined features."""
        vowels, consonants = self._parse_phonemes(phonemes)

        v_string = self._phoneme_set_to_string(vowels)
        c_string = self._phoneme_set_to_string(consonants)

        v_descriptors = self._grouped_phoneme_descriptors(vowels)
        c_descriptors = self._grouped_phoneme_descriptors(consonants)

        all_descriptors = defaultdict(set)
        for d in v_descriptors:
            for k, v in d.items():
                all_descriptors[k].add(v)
        for d in c_descriptors:
            for k, v in d.items():
                all_descriptors[k].add(v)

        groups_to_exclude = set()
        for k, v in all_descriptors.items():
            # There are features which are not in the predefined feature space
            # So we exclude the whole group.
            if all_descriptors[k] - self.feature_names:
                groups_to_exclude.add(k)

        self.groups_excluded = groups_to_exclude

        v_descriptors = [{k: v for k, v in d.items()
                         if k not in groups_to_exclude} for d in v_descriptors]

        c_descriptors = [{k: v for k, v in d.items()
                         if k not in groups_to_exclude} for d in c_descriptors]

        vowel_vectors = []
        for descriptors in v_descriptors:
            vec = []
            for k, v in sorted(descriptors.items()):
                if v != 'absent':
                    vec.extend(self.phoneme_features[v])
                else:
                    vec.append(0)
            vowel_vectors.append(vec)

        consonant_vectors = []

        for descriptors in c_descriptors:
            vec = []
            for k, v in sorted(descriptors.items()):
                if v != 'absent':
                    vec.extend(self.phoneme_features[v])
                else:
                    vec.append(0)
            consonant_vectors.append(vec)

        vowel_dict = dict(zip(v_string, vowel_vectors))
        consonant_dict = dict(zip(c_string, consonant_vectors))

        return vowel_dict, consonant_dict
