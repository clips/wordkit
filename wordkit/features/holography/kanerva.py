"""The kanerva method of holographic representation."""
import numpy as np

from .base import (HolographicTransformer,
                   OpenNGramMixIn,
                   NGramMixIn,
                   LinearMixIn,
                   ConstrainedOpenNGramMixIn)


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

    def inverse_transform(self, X, threshold=.25):
        if np.ndim(X) == 1:
            X = X[None, :]
        words = []
        letters, vecs = zip(*self.features.items())
        vecs = np.stack(vecs)
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        for x in X:
            w = []
            for inv_pos in self.inv:
                dec = x[inv_pos]
                dec /= np.linalg.norm(dec)
                sim = dec.dot(vecs.T)
                if sim.max() > threshold:
                    w.append(letters[sim.argmax()])
            words.append("".join(w))
        return words


class KanervaNGramTransformer(KanervaTransformer, NGramMixIn):

    def __init__(self, vec_size, n, use_padding=True, density=1.0, field=None):
        super().__init__(vec_size, density, field)
        self.n = n
        self.use_padding = use_padding


class KanervaLinearTransformer(KanervaTransformer, LinearMixIn):
    pass


class KanervaOpenNGramTransformer(KanervaTransformer, OpenNGramMixIn):

    def __init__(self, vec_size, n, density=1.0, field=None):
        super().__init__(vec_size, density, field)
        self.n = n


class KanervaConstrainedOpenNGramTransformer(KanervaTransformer,
                                             ConstrainedOpenNGramMixIn):

    def __init__(self,
                 vec_size,
                 n,
                 window,
                 use_padding=True,
                 density=1.0,
                 field=None):
        super().__init__(vec_size, density, field)
        self.n = n
        self.window = window
        self.use_padding = use_padding
