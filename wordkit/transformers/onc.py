"""Featurization based on Onset Nucleus Coda."""
import numpy as np
import re

from .base import FeatureTransformer
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
            a tuple of a dictionary of features, for vowels and consonants.
            an initialized FeatureExtractor instance.

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

        self.phon_indexer = []
        self.grid_indexer = []

        self.num_syls = 0
        self.o = 0
        self.n = 0
        self.c = 0
        self.vec_len = 0
        self.syl_len = 0
        self.grid = ""

        # Regex for detecting consecutive occurrences of the letter V
        self.r = re.compile(r"V+")

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
        self.num_syls = num_syls

        grid = ["C" * self.o, "V" * self.n, "C" * self.c]
        self.grid = "".join(chain.from_iterable(grid * self.num_syls))
        self.syllable_grid = grid

        self.phon_indexer = []
        self.grid_indexer = []
        idx = 0
        idx_2 = 0
        for i in grid:
            if i == "C":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.consonant_length
                idx_2 += len(self.consonant2idx)
            elif i == "V":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.vowel_length
                idx_2 += len(self.vowel2idx)

    def _fit(self, X):
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
        vowels, consonants = self.features
        if " " not in vowels:
            vowels[" "] = np.zeros_like(list(vowels.values())[0])
        if " " not in consonants:
            consonants[" "] = np.zeros_like(list(consonants.values())[0])

        self.vowels = vowels
        self.consonants = consonants
        self.phonemes = set(vowels.keys())
        self.phonemes.update(consonants.keys())
        self.feature_names = self.phonemes

        self.vowel_length = len(next(self.vowels.values().__iter__()))
        self.consonant_length = len(next(self.consonants.values().__iter__()))

        self.idx2consonant = {idx: c
                              for idx, c in enumerate(sorted(self.consonants))}
        self.consonant2idx = {v: k for k, v in self.idx2consonant.items()}
        self.idx2vowel = {idx: v
                          for idx, v in enumerate(sorted(self.vowels))}
        self.vowel2idx = {v: k for k, v in self.idx2vowel.items()}
        self.phoneme2idx = {p: idx
                            for idx, p in enumerate(sorted(self.phonemes))}

        if type(X[0]) == dict:
            X = [x[self.field] for x in X]
        self._check(chain.from_iterable(X))

        num_syls = max([len(x) for x in X])

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

        self._set_grid_params((o, n, c), num_syls)
        self._is_fit = True

        return self

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
        self._check(x)
        if not self._is_fit:
            raise ValueError("The vectorizer has not been fit yet.")
        if len(x) > self.num_syls:
            raise ValueError("{0} is too long".format(x))

        vec = []

        for idx in range(self.num_syls):

            # Define a syllable zero vector
            syll_vec = np.zeros(self.syl_len,)
            # The vector index at which the nucleus starts
            n_idx = self.o * self.consonant_length
            # The vector index at which the coda starts
            c_idx = (self.n * self.vowel_length) + n_idx

            try:
                s = x[idx]
            except IndexError:
                # If the current word does not have syllable here,
                # append the zero vector.
                vec.append(syll_vec)
                continue

            # Create CVC grid from phoneme representation
            cvc = "".join(["C" if p in self.consonants else "V" for p in s])
            try:
                m = next(self.r.finditer(cvc))
                # Letter index of the nucleus
                n = m.start()
                # Letter index of the coda
                c = len(m.group()) + n

                for lidx, p in enumerate(s[:n]):
                    start = lidx * self.consonant_length
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[p]

                for lidx, p in enumerate(s[n:c]):
                    start = (lidx * self.vowel_length) + n_idx
                    end = start + self.vowel_length
                    syll_vec[start: end] = self.vowels[p]

                for lidx, p in enumerate(s[c:]):
                    start = (lidx * self.consonant_length) + c_idx
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[p]

            except StopIteration:

                for lidx, p in enumerate(s[:len(cvc)]):
                    start = (lidx * self.consonant_length)
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[p]

            vec.append(syll_vec)

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
        for word in reshaped:
            yield tuple([tuple(x) for x in
                        [[p for p in x if p != " "]
                        for x in word] if x])
