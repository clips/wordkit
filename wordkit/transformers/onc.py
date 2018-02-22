"""Featurization based on Onset Nucleus Coda."""
import numpy as np
import re

from .base import FeatureTransformer
from copy import copy
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
    ==========
    features : tuple of dictionaries
        A tuple of dictionaries, containing vowel and consonant features,
        respectively.

    grid : tuple of triples, optional, default ()
        Containing the number of ONC clusters, and the number of O, N, and C
        components of said clusters. Thus, if the user passes
        ((4, 2, 3), (2, 1, 2)), the ONCTransformer will have 2 clusters, with
        the first onset being 4 consonants long, the first nucleus being 2
        vowels and the first coda being 3 consonants.
        If this is left blank, the optimal grid is calculated automatically.
        If you leave the grid blank, there is the distinct possibility that
        any held out data can not be fitted with this transformer, so beware.

    """

    def __init__(self, features, grid=()):
        """Encode syllables with Onset Nucleus Coda encoding."""
        super().__init__(features, "syllables")

        vowels, consonants = features
        if " " not in vowels:
            vowels[" "] = np.zeros_like(list(vowels.values())[0])
        if " " not in consonants:
            consonants[" "] = np.zeros_like(list(consonants.values())[0])

        self.vowels = vowels
        self.consonants = consonants
        self.features = copy(vowels)
        self.features.update(consonants)

        self.vowel_length = len(next(self.vowels.values().__iter__()))
        self.consonant_length = len(next(self.consonants.values().__iter__()))

        self.idx2consonant = {idx: c for idx, c in enumerate(self.consonants)}
        self.consonant2idx = {v: k for k, v in self.idx2consonant.items()}
        self.idx2vowel = {idx: v for idx, v in enumerate(self.vowels)}
        self.vowel2idx = {v: k for k, v in self.idx2vowel.items()}
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.features)}

        self._is_fit = False

        self.phon_indexer = []
        self.grid_indexer = []

        if grid:
            self._set_grid_params(grid)
            self._is_fit = True

        else:
            self.num_syls = 0
            self.o = 0
            self.n = 0
            self.c = 0
            self.vec_len = 0
            self.syl_len = 0

        # Regex for detecting consecutive occurrences of the letter V
        self.r = re.compile(r"V+")

    def grid(self):
        """Extract the grid from a fit instance."""
        if not self._is_fit:
            raise ValueError("The vectorizer has not been fit yet. "
                             "Hence, it does not have a grid to extract.")
        grid = [(self.o[idx], self.n[idx], self.c[idx]) for
                idx in range(self.num_syls)]

        return grid

    def _set_grid_params(self, grid):
        """
        Set the grid params given a grid.

        This function is meanth to remove duplication between the transform
        function and instantiation through passing a grid.

        Parameters
        ==========
        grid : tuple of triples:
            A tuple of triples describing the grid clusters. See __init__
            for more documentation.

        """
        self.o, self.n, self.c = [np.array(x) for x in zip(*grid)]
        self.syl_len = (self.o * self.consonant_length)
        self.syl_len += (self.n * self.vowel_length)
        self.syl_len += (self.c * self.consonant_length)
        self.vec_len = self.syl_len.sum()
        self.num_syls = len(grid)

        grid = []

        for idx in range(self.num_syls):
            for n, cvc in zip([self.o[idx], self.n[idx], self.c[idx]], "CVC"):
                grid.extend(n * cvc)

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

    def fit(self, X):
        """
        Calculate the best Onset Nucleus Coda grid given X.

        This function calculates the ideal Onset Nucleus Coda grid, that is,
        the number of consonants and vowels in each part of the grid necessary
        to describe the corpus. The number of phonemes per cluster can differ
        between syllables.


        Parameters
        ==========
        X : a list of syllable tuples.
            Represents the words to fit the ONCTransformer on.

        Returns
        =======
        self : ONCTransformer
            Return a fitted ONCTransformer

        """
        if type(X[0]) == dict:
            X = [x[self.field] for x in X]
        self._check(chain.from_iterable(X))

        num_syls = max([len(x) for x in X])

        o = np.zeros(num_syls, dtype=np.int32)
        n = np.zeros(num_syls, dtype=np.int32)
        c = np.zeros(num_syls, dtype=np.int32)

        for syll in X:

            for idx, cvc in enumerate(syll):
                cvc = "".join(["C" if x in self.consonants
                               else "V" for x in cvc])
                c_l = len(cvc)
                try:
                    m = next(self.r.finditer(cvc))
                    o[idx] = max(m.start(), o[idx])
                    n[idx] = max(len(m.group()), n[idx])
                    c[idx] = max(c_l - m.end(), c[idx])
                except StopIteration:
                    o[idx] = max(c_l, o[idx])
        grid = []
        for idx in range(num_syls):
            grid.append([o[idx], n[idx], c[idx]])

        self._set_grid_params(grid)
        self._is_fit = True

        return self

    def vectorize(self, x):
        """
        Vectorize a single word.

        This function converts a single list of syllable strings to a feature
        vector. In order to use this function, the vectorizer must have been
        fit first, either by using the fit function, or by passing a
        pre-defined grid.

        Parameters
        ==========
        x : A string or dictionary with 'syllables' as key.
            The word to vectorize

        Returns
        =======
        v : numpy array
            The vectorized word.

        """
        if type(x) == dict:
            phonemes = x[self.field]
        else:
            phonemes = x
        self._check(phonemes)
        if not self._is_fit:
            raise ValueError("The vectorizer has not been fit yet.")
        if len(phonemes) > self.num_syls:
            raise ValueError("{0} is too long".format(phonemes))

        vec = []

        for idx in range(self.num_syls):

            # Define a syllable zero vector
            syll_vec = np.zeros(self.syl_len[idx])
            # The vector index at which the nucleus starts
            n_idx = self.o[idx] * self.consonant_length
            # The vector index at which the coda starts
            c_idx = (self.n[idx] * self.vowel_length) + n_idx

            try:
                s = phonemes[idx]
            except IndexError:
                # If the current word does not have syllable here,
                # append the zero vector.
                vec.append(syll_vec)
                continue

            # Create CVC grid from phoneme representation
            cvc = "".join(["C" if x in self.consonants else "V" for x in s])
            try:
                m = next(self.r.finditer(cvc))
                # Letter index of the nucleus
                n = m.start()
                # Letter index of the coda
                c = len(m.group()) + n

                for lidx, x in enumerate(s[:n]):
                    start = lidx * self.consonant_length
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[x]

                for lidx, x in enumerate(s[n:c]):
                    start = (lidx * self.vowel_length) + n_idx
                    end = start + self.vowel_length
                    syll_vec[start: end] = self.vowels[x]

                for lidx, x in enumerate(s[c:]):
                    start = (lidx * self.vowel_length) + c_idx
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[x]

            except StopIteration:

                for lidx, x in enumerate(s[:len(cvc)]):
                    start = (lidx * self.consonant_length)
                    end = start + self.consonant_length
                    syll_vec[start: end] = self.consonants[x]

            vec.append(syll_vec)

        return np.concatenate(vec).ravel()

    def inverse_transform(self, X):
        """Transform a matrix back into their word representations."""
        # TODO: fix error if len(vowels) == len(consonants)
        # TODO: use grid indexer here.
        vowel_keys, vowels = zip(*self.vowels.items())
        consonant_keys, consonants = zip(*self.consonants.items())

        vowels = np.array(vowels)
        consonants = np.array(consonants)

        ends = self.grid_indexer[1:] + [self.vec_len]

        words = []

        for x in X:
            word = []
            for b, e in zip(self.grid_indexer, ends):
                if e - b == vowels.shape[1]:
                    diff = x[b:e] - vowels
                    res = np.linalg.norm(diff, axis=-1).argmin()
                    word.append(vowel_keys[res])
                else:
                    diff = x[b:e] - consonants
                    res = np.linalg.norm(diff, axis=-1).argmin()
                    word.append(consonant_keys[res])

            words.append(tuple([x for x in word if x != " "]))

        return words
