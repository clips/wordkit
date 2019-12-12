"""The kanerva method of holographic representation."""
import numpy as np

from .base import (HolographicTransformer,
                   NGramMixin,
                   OpenNgramMixin,
                   ConstrainedOpenNgramMixin,
                   LinearMixin)


class KanervaTransformer(HolographicTransformer):

    def __init__(self, vec_size, field=None):
        super().__init__(vec_size, field)
        assert (vec_size % 2) == 0

    def generate(self, size):
        assert len(size) == 2
        vecs = np.zeros(size)
        for x in vecs:
            idx = np.random.permutation(len(x))
            high, low = idx.reshape(2, -1)
            x[high] = 1

        return vecs.astype(np.bool)

    def generate_positions(self, size):
        return self.generate(size)

    @staticmethod
    def xor(item, idx):
        return item ^ idx

    def compose(self, item, idx):
        item = self.features[item]
        idx = self.positions[idx]
        return self.xor(item, idx)

    def add(self, X):
        return np.mean(X, 0) > .5

    def inverse_transform(self, X, threshold=.15):
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
                if sim.max() > threshold:
                    w.append(letters[sim.argmax()])
            words.append("".join(w))
        return words


class KanervaNGramTransformer(KanervaTransformer, NGramMixin):

    def __init__(self, vec_size, n, use_padding=True, field=None):
        super().__init__(vec_size, field)
        self.n = n
        self.use_padding = use_padding
        self.field = field


class KanervaLinearTransformer(KanervaTransformer, LinearMixin):

    pass


class KanervaOpenNGramTransformer(KanervaTransformer, OpenNgramMixin):

    def __init__(self, vec_size, n, field=None):
        super().__init__(vec_size, field)
        self.n = n


class KanervaConstrainedOpenNGramTransformer(KanervaTransformer,
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
