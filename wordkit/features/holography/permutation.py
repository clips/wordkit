"""The plate method of holographic representation."""
import numpy as np

from .base import (HolographicTransformer,
                   LinearMixin,
                   ConstrainedOpenNgramMixin,
                   NGramMixin,
                   OpenNgramMixin)


class PermutationTransformer(HolographicTransformer):

    def generate(self, size):
        return np.random.normal(size=size)

    def generate_positions(self, size):
        num, size = size
        return np.stack([np.random.permutation(size) for x in range(num)])

    def compose(self, item, idx):
        item = self.features[item]
        idx = self.positions[idx]
        return item[idx]

    def decompose(self, vec, idx):
        idx = self.positions[idx]
        return vec[np.argsort(idx)]

    def add(self, X):
        return np.sum(X, 0)

    def inverse_transform(self, X, threshold=.25):
        if np.ndim(X) == 1:
            X = X[None, :]
        words = []
        letters, vecs = zip(*self.features.items())
        vecs = np.stack(vecs)
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        for x in X:
            w = []
            for _, v in sorted(self.positions.items()):
                dec = self.decompose(x, v)
                dec /= np.linalg.norm(dec)
                sim = dec.dot(vecs.T)
                if sim.max() > threshold:
                    w.append(letters[sim.argmax()])
            words.append("".join(w))
        return words


class PermutationNGramTransformer(PermutationTransformer, NGramMixin):

    def __init__(self, vec_size, n, use_padding=True, field=None):
        super().__init__(vec_size, field)
        self.n = n
        self.use_padding = use_padding
        self.field = field


class PermutationLinearTransformer(PermutationTransformer, LinearMixin):

    pass


class PermutationOpenNGramTransformer(PermutationTransformer, OpenNgramMixin):

    def __init__(self, vec_size, n, field=None):
        super().__init__(vec_size, field)
        self.n = n


class PermutationConstrainedOpenNGramTransformer(PermutationTransformer,
                                                 ConstrainedOpenNgramMixin):

    def __init__(self,
                 vec_size,
                 n,
                 window,
                 use_padding=True,
                 field=None):
        super().__init__(vec_size, field)
        self.n = n
        self.window = window
        self.use_padding = use_padding
