"""The plate method of holographic representation."""
import numpy as np

from .base import (HolographicTransformer,
                   OpenNGramMixIn,
                   NGramMixIn,
                   LinearMixIn,
                   ConstrainedOpenNGramMixIn)


class PlateTransformer(HolographicTransformer):

    def generate(self, size):
        return np.random.normal(size=size, scale=1/size[1])

    def generate_positions(self, size):
        return self.generate(size)

    def compose(self, item, idx):
        item = self.features[item]
        idx = self.positions[idx]
        return self.circular_convolution(item, idx)

    @staticmethod
    def circular_convolution(x, y):
        return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real

    def add(self, X):
        return np.sum(X, 0)

    @staticmethod
    def involution(x):
        return np.concatenate([x[None, 0], x[-1:0:-1]])

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
                dec = self.circular_convolution(self.involution(v), x)
                dec /= np.linalg.norm(dec)
                sim = dec.dot(vecs.T)
                if sim.max() > threshold:
                    w.append(letters[sim.argmax()])
            words.append("".join(w))
        return words


class PlateNGramTransformer(PlateTransformer, NGramMixIn):

    def __init__(self, vec_size, n, use_padding=True, field=None):
        super().__init__(vec_size, field)
        self.n = n
        self.use_padding = use_padding


class PlateLinearTransformer(PlateTransformer, LinearMixIn):
    pass


class PlateOpenNGramTransformer(PlateTransformer, OpenNGramMixIn):

    def __init__(self, vec_size, n, field=None):
        super().__init__(vec_size, field)
        self.n = n


class PlateConstrainedOpenNGramTransformer(PlateTransformer,
                                           ConstrainedOpenNGramMixIn):

    def __init__(self, vec_size, n, window, use_padding=True, field=None):
        super().__init__(vec_size, field)
        self.n = n
        self.window = window
        self.use_padding = use_padding
