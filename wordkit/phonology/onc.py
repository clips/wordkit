"""Featurization based on Onset Nucleus Coda."""
import numpy as np
import re

from ..base.transformer import FeatureTransformer
from itertools import chain


class ONCTransformer(FeatureTransformer):
    """
    Encode syllables with Onset Nucleus Coda encoding.

    ONC encodes words as Onset Nucleus Coda feature structures by assuming
    a CVC structure for each syllable. The number of phonemes per element
    is variable, and will be determined based on some data, which ensures
    a parsimonious representation.

    The input to the constructor is two dictionaries, where the keys are
    phonemes as unicode characters, and the values are feature arrays for
    these phonemes. ONC is thus completely agnostic to actual feature
    system used.

    This transformer can not handle orthographic fields.

    Parameters
    ----------
    features : tuple of dicts, or FeatureExtractor instance.
        features can either be
            - a tuple of a dictionary of features, for vowels and consonants.
            - an initialized FeatureExtractor instance.

        In the first case, the features you input to the Transformer are
        used. In the final case, the FeatureExtractor is used to extract
        features from your input during fitting.

        The choice between pre-defined featues and an is purely a matter of
        convenience. First extracting features using the FeatureExtractor
        leads to the same result as using the FeatureExtractor directly.

    """

    def __init__(self, features, field=None):
        """Encode syllables with Onset Nucleus Coda encoding."""
        super().__init__(features, field=field)

        self._is_fit = False

        self.o = 0
        self.n = 0
        self.c = 0
        self.vec_len = 0
        self.syl_len = 0
        self.consonant_length = 0
        self.vowel_length = 0
        self.syl_len = 0
        self.num_syls = 0

        # Regex for detecting consecutive occurrences of the letter V
        self.r = re.compile(r"V+")

    @property
    def num_syls(self):
        """The number of syllables."""
        return self.__num_syls

    @num_syls.setter
    def num_syls(self, value):
        """Makes sure the grid parameters are recalculated."""
        self.__num_syls = value
        self._set_grid_params((self.o, self.n, self.c), value)

    def _set_grid_params(self, grid, num_syls):
        """
        Set the grid params given a grid.

        Parameters
        ----------
        grid : tuple of triples
            A tuple of triples describing the grid clusters. See __init__
            for more documentation.
        num_syls : int
            The number of syllables to use.

        """
        self.o, self.n, self.c = grid
        self.syl_len = (self.o * self.consonant_length)
        self.syl_len += (self.n * self.vowel_length)
        self.syl_len += (self.c * self.consonant_length)
        self.vec_len = self.syl_len * num_syls

        grid = ["C" * self.o, "V" * self.n, "C" * self.c]
        self.grid = "".join(chain.from_iterable(grid * self.num_syls))
        self.syllable_grid = grid

    def _validate(self, X):
        """
        Check whether an input dataset contains illegal features.

        Calculate the difference of the keys of the feature dict and x.
        Raises a ValueError if the result is non-empty.

        Parameters
        ----------
        X : list of strings or list of dicts.
            An input dataset.

        """
        # Weird transform because of the nesting in syllables.
        feats = set(chain.from_iterable([chain.from_iterable(x) for x in X]))
        overlap = feats.difference(self.feature_names)
        if overlap:
            raise ValueError("The sequence contained illegal features: {0}"
                             .format(overlap))

    def fit(self, X):
        """
        Calculate the best Onset Nucleus Coda grid given X.

        This function calculates the ideal Onset Nucleus Coda grid, that is,
        the number of consonants and vowels in each part of the grid necessary
        to describe the corpus.

        Parameters
        ----------
        X : a list of syllable tuples.
            Represents the words to fit the ONCTransformer on.

        Returns
        -------
        self : ONCTransformer
            Return a fitted ONCTransformer

        """
        super().fit(X)
        vowels, consonants = self.features

        self.vowels = vowels
        self.consonants = consonants
        self.phonemes = set(vowels.keys())
        self.phonemes.update(consonants.keys())
        self.feature_names = self.phonemes

        self.vowel_length = len(next(self.vowels.values().__iter__()))
        self.consonant_length = len(next(self.consonants.values().__iter__()))

        X = self._unpack(X)
        self._validate(X)

        self.num_syls = max([len(x) for x in X])

        o = 0
        n = 0
        c = 0

        for syll in set(chain.from_iterable(X)):

            cvc = "".join(["C" if x in self.consonants
                           else "V" for x in syll])
            c_l = len(cvc)
            try:
                m = next(self.r.finditer(cvc))
                o = max(m.start(), o)
                n = max(len(m.group()), n)
                c = max(c_l - m.end(), c)
            except StopIteration:
                c = max(c_l, c)

        self._set_grid_params((o, n, c), self.num_syls)
        self._is_fit = True

        return self

    def put_on_grid(self, x):
        """Put phonemes on a syllabic grid."""
        grid = []

        m = next(self.r.finditer(self.grid))
        # Start index of the nucleus
        grid_n = m.start()
        # Letter index of coda.
        grid_c = len(m.group()) + grid_n
        for s in x:
            syll = [" " for x in "".join(self.syllable_grid)]
            # s is a syllable.
            cvc = "".join(["C" if p in self.consonants else "V" for p in s])
            try:
                m = next(self.r.finditer(cvc))
                # Letter index of the nucleus
                n = m.start()
                # Letter index of the coda
                c = len(m.group()) + n
                for idx in range(n):
                    syll[idx] = s[idx]
                for idx in range(c-n):
                    syll[grid_n + idx] = s[n+idx]
                for idx in range(0, len(cvc) - c):
                    syll[grid_c + idx] = s[c+idx]
            except StopIteration:
                syll[-len(cvc):] = s
            grid.extend(syll)
        empty_grid = [" " for x in "".join(self.syllable_grid)]
        grid.extend((self.num_syls - len(x)) * empty_grid)
        return grid

    def vectorize(self, x):
        """
        Vectorize a single word.

        This function converts a single list of syllable strings to a feature
        vector. In order to use this function, the vectorizer must have been
        fit first.

        Parameters
        ----------
        x : A string or dictionary with 'syllables' as key
            The word to vectorize

        Returns
        -------
        v : numpy array
            The vectorized word.

        """
        if not self._is_fit:
            raise ValueError("The vectorizer has not been fit yet.")
        if len(x) > self.num_syls:
            raise ValueError("{0} is too long".format(x))

        grid_form = self.put_on_grid(x)
        vec = []
        for x, cv in zip(grid_form, self.grid):
            try:
                if cv == "C":
                    vec.append(self.consonants[x])
                else:
                    vec.append(self.vowels[x])
            except KeyError:
                if cv == "C":
                    vec_len = next(iter(self.consonants.values()))
                else:
                    vec_len = next(iter(self.vowels.values()))
                vec.append(np.zeros_like(vec_len))

        return np.concatenate(vec).ravel()

    def inverse_transform(self, X):
        """Transform a matrix back into their word representations."""
        vowel_keys, vowels = zip(*self.vowels.items())
        consonant_keys, consonants = zip(*self.consonants.items())

        vowels = np.array(vowels)
        consonants = np.array(consonants)

        idx = 0

        words = []
        for x in self.grid:
            if x == "C":
                s = consonants
                s_k = consonant_keys
            else:
                s = vowels
                s_k = vowel_keys
            diff = X[:, idx:idx+s.shape[1]][:, None, :] - s[None, :, :]
            indices = np.linalg.norm(diff, axis=-1).argmin(-1)
            words.append([s_k[x] for x in indices])
            idx += s.shape[1]

        reshaped = np.array(words).T.reshape(X.shape[0], self.num_syls, -1)
        return [tuple([tuple(x) for x in
                      [[p for p in x if p != " "]
                      for x in word] if x]) for word in reshaped]

    def list_features(self, words):
        """List the features."""
        if not self._is_fit:
            raise ValueError("Transformer has not been fit yet.")
        words = self._unpack(words)
        indices = range(len(self.grid))
        return [list(zip(self.put_on_grid(x), indices)) for x in words]
