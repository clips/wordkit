"""The kanerva method of holographic representation."""
import numpy as np

from .base import (HolographicTransformer,
                   OpenNGramMixIn,
                   NGramMixIn,
                   LinearMixIn)


class KanervaTransformer(HolographicTransformer):

    def __init__(self, vec_size, density=1.0, field=None):
        super().__init__(vec_size, field)
        assert .0 < density <= 1.0
        assert (vec_size % 2) == 0
        density = int(vec_size * density)
        density += density % 2
        self.density = density

    def generate(self, size):
        assert len(size) == 2
        vecs = np.zeros(size)
        for x in vecs:
            idx = np.random.permutation(len(x))[:self.density]
            high, low = idx.reshape(2, -1)
            x[high] = 1
            x[low] = -1

        assert np.all(vecs.sum(1) == 0)

        return vecs

    def generate_positions(self, size):
        assert len(size) == 2
        num, size = size
        p = np.stack([np.random.permutation(size) for x in range(num)])
        self.inv = np.stack([np.argsort(x) for x in p])
        return p

    def compose(self, item, idx):
        return self.features[item][self.positions[idx]]

    def add(self, a, b):
        return a + b


class KanervaNGramTransformer(KanervaTransformer, NGramMixIn):

    def __init__(self, vec_size, n, density=1.0):
        super().__init__(vec_size, density)
        self.n = n


class KanervaLinearTransformer(KanervaTransformer, LinearMixIn):
    pass


class KanervaOpenNGramTransformer(KanervaTransformer, OpenNGramMixIn):

    def __init__(self, vec_size, n, density=1.0):
        super().__init__(vec_size, density)
        self.n = n
