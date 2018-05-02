"""Description."""
import numpy as np

from .base import FeatureTransformer


class CVTransformer(FeatureTransformer):
    r"""
    Put phonemes on a Consonant Vowel Grid.

    This class is a generalization of the Patpho system.
    Patpho is a system for converting sequences of phonemes to vector
    representations that capture phonological similarity of words.

    If you use the CVTransformer, you _must_ cite the following publication.

    @article{li2002patpho,
      title={PatPho: A phonological pattern generator for neural networks},
      author={Li, Ping and MacWhinney, Brian},
      journal={Behavior Research Methods, Instruments, \& Computers},
      volume={34},
      number={3},
      pages={408--415},
      year={2002},
      publisher={Springer}
    }

    The original C implementation can be found here (October 2017):
    http://www.personal.psu.edu/pul8/patpho_e.shtml

    Note that our implementation also supports the conversion of orthographic
    strings using consonant vowel representations.

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

    grid_structure : string, default="CCCVV"
        A list describing the Consonant Vowel structure of the
        CVTransformer. During fitting, the CVTransformer determines how
        many of these grid clusters are necessary to fit the entire
        dataset.

    left : bool
        Whether to use right- or left-justified encoding when placing
        phonemes on the grid.

    field : str, default 'phonology'
        The field to use.

    """

    def __init__(self,
                 features,
                 grid_structure="CCCVV",
                 left=True,
                 field='phonology'):
        """Put phonemes on a consonant vowel grid."""
        super().__init__(features, field)
        self.grid_structure = grid_structure
        self.left = left

    def init_grid(self):
        """
        Initialize the syllabic grid.

        The syllabic grid consists of the grid, as defined in the constructor,
        * the maximum allowed length. The grid always closes with a number
        of consonant clusters, equal to the number of consonant clusters in
        the grid.

        returns
        -------
        grid : list
            A list of 'C' and 'V' strings, representing
            whether that position in the grid represents a vowel
            or a consonant.

        """
        first_v = self.grid_structure.index("V")
        grid = self.grid_structure * self.max_length
        grid += self.grid_structure[first_v]

        return grid

    def _fit(self, X):
        """
        Fit the CVTransformer to find the optimal number of grids required.

        Parameters
        ----------
        X : list of phoneme strings, or a list of dictionaries.
            The words you want to fit the CVTransformer on.

        Returns
        -------
        self : CVTransformer
            The CVTransformer.

        """
        if type(X[0]) == dict:
            X = [x[self.field] for x in X]

        vowels, consonants = self.features
        if " " not in vowels:
            vowels[" "] = np.zeros_like(list(vowels.values())[0])
        if " " not in consonants:
            consonants[" "] = np.zeros_like(list(consonants.values())[0])
        self.phonemes = set(vowels.keys())
        self.phonemes.update(consonants.keys())
        self.feature_names = self.phonemes
        self.vowel_length = len(list(vowels.values())[0])
        self.consonant_length = len(list(consonants.values())[0])

        if any([len(v) != self.vowel_length for v in vowels.values()]):
            raise ValueError("Not all vowel vectors have the same length")

        if any([len(v) != self.consonant_length for v in consonants.values()]):
            raise ValueError("Not all consonant vectors have the same length")

        # consonant dictionary
        self.consonants = consonants
        # vowel dictionary
        self.vowels = vowels
        self._check(X)

        # indexes
        self.idx2consonant = {idx: c
                              for idx, c in enumerate(sorted(self.consonants))}
        self.consonant2idx = {v: k for k, v in self.idx2consonant.items()}
        self.idx2vowel = {idx: v
                          for idx, v in enumerate(sorted(self.vowels))}
        self.vowel2idx = {v: k for k, v in self.idx2vowel.items()}
        self.phoneme2idx = {p: idx
                            for idx, p in enumerate(sorted(self.phonemes))}

        first_v = self.grid_structure.index("V")
        self.grid = self.grid_structure + self.grid_structure[:first_v]
        last_v = -first_v
        idx = 0

        while idx < len(X):
            try:
                self._put_on_grid(X[idx])
                idx += 1
            except IndexError:
                self.grid = self.grid[:last_v] + self.grid_structure
                self.grid += self.grid_structure[:first_v]

        # Add the number of consonants to the end of the grid structure

        # The grid indexer contains the number of _Features_
        # the grid contains up to that point.
        self.grid_indexer = []

        # The phon indexer contains the number of phonemes
        # the grid contains up to that point.
        self.phon_indexer = []
        idx = 0
        idx_2 = 0
        for i in self.grid:
            if i == "C":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.consonant_length
                idx_2 += len(self.consonants)
            elif i == "V":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.vowel_length
                idx_2 += len(self.vowels)

        # The total length of the grid in features
        self.vec_len = idx

        self._is_fit = True
        return self

    def vectorize(self, x):
        """
        Convert a phoneme sequence to a vector representation.

        Parameters
        ----------
        x : a single phoneme sequence or dictionary with 'phonology' as a key.
            The phoneme sequence.

        Returns
        -------
        v : numpy array
            A vectorized version of the phoneme sequence.

        """
        grid = self._put_on_grid(x)

        # convert syllabic grid to vector
        phon_vector = np.zeros(self.vec_len)

        for idx, phon in enumerate(grid):
            p = self.phonemes[phon]
            g_idx = self.grid_indexer[idx]
            phon_vector[g_idx: g_idx+len(p)] = p

        return np.array(phon_vector)

    def _put_on_grid(self, x):
        """Put phonemes on a grid."""
        if not self.left:
            x = x[::-1]

        word_grid = ["C" if p in self.consonants else "V" for p in x]

        word_idx = 0
        grid_idx = 0

        indices = [" "] * len(self.grid)

        while word_idx < len(word_grid):
            if word_grid[word_idx] == self.grid[grid_idx]:
                indices[grid_idx] = x[word_idx]
                word_idx += 1
                grid_idx += 1
            else:
                grid_idx += 1
            if grid_idx > len(self.grid):
                raise IndexError("Too long")

        if not self.left:
            indices = indices[::-1]

        return indices

    def grid_indices(self, x):
        """
        Get the grid indices for a given phoneme input.

        Useful for determining whether two inputs overlap
        in their consonants and vowels (as opposed to their
        features)
        """
        overlap = self._check(x)
        if overlap:
            raise ValueError("{0} contains invalid phonemes: {1}"
                             .format(x, " ".join(overlap)))

        grid = self._put_on_grid(x)
        grid = [self.phoneme2idx[v] + self.phon_indexer[idx]
                for idx, v in enumerate(grid)]
        return sorted(grid)

    def inverse_transform(self, X):
        """Transform a set of word representations back to their form."""
        vowel_keys, vowels = zip(*self.vowels.items())
        consonant_keys, consonants = zip(*self.consonants.items())

        vowels = np.array(vowels)
        consonants = np.array(consonants)

        words = []

        idx = 0
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

        reshaped = np.array(words).T
        return tuple([tuple([z for z in x if z != " "]) for x in reshaped])
