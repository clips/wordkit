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

        return vecs

    def generate_positions(self, size):
        return self.generate(size)

    @staticmethod
    def xor(item, idx):
        return (((item * idx) == -1) * 2) - 1

    def compose(self, item, idx):
        item = self.features[item]
        idx = self.positions[idx]
        return self.xor(item, idx)

    def add(self, X):
        return ((np.mean(X, 0) > 0) * 2) - 1

    def inverse_transform(self, X, threshold=.25):
        if np.ndim(X) == 1:
            X = X[None, :]
        words = []
        letters, vecs = zip(*self.features.items())
        vecs = np.stack(vecs).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        for x in X:
            w = []
            for pos in self.positions.values():
                dec = self.xor(x, pos).astype(np.float32)
                dec /= np.linalg.norm(dec)
                sim = dec.dot(vecs.T)
                print(sim)
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
