"""Sampler class."""
import numpy as np
from sklearn.base import TransformerMixin


class Sampler(TransformerMixin):
    """
    Sample from a list of words, based on the frequency of those words.

    Parameters
    ----------
    X : numpy array
        Your vectorized data.

    words : list of strings.
        The identities of the words.

    frequencies : numpy array or list of integers, default None.
        The frequencies of your input data. If this is None, the Sampler
        will sample uniformly.

    smoothing : bool
        Whether to smooth the distribution by adding 1 to frequency counts.
        If mode is set to 'log', the frequencies are always smoothed to avoid
        log(0).

    mode : string {'raw', 'log'}, default 'raw'
        The mode to use for frequency to probability conversion.
        'raw' uses the raw frequency counts, while 'log' uses log scaling.

    replacement : bool
        Whether to sample with or without replacement.

    """

    def __init__(self,
                 X,
                 words,
                 frequencies=None,
                 smoothing=True,
                 mode='raw',
                 replacement=True):
        """Sample from a distribution over words."""
        self.smoothing = smoothing
        self.mode = mode
        if frequencies is None:
            frequencies = np.ones(X.shape[0])
        else:
            frequencies = np.asarray(frequencies)
            if self.mode == 'log':
                frequencies = np.log(frequencies + 1)
            elif self.smoothing:
                frequencies += 1

        self.X = X
        self.words = words
        self.frequencies = frequencies / np.sum(frequencies)
        self.replacement = replacement

    def sample(self, num_to_sample):
        """
        Sample from the featurized data.

        Parameters
        ----------
        num_to_sample : int
            The number of items to sample.

        Returns
        -------
        features : np.array
            A matrix of sampled data.

        words : tuple
            The sampled words.

        """
        if not self.replacement and num_to_sample > len(self.X):
            raise ValueError("Your tried to sample without replacement from "
                             "a set which is smaller than your sample size "
                             ": sample size: {} set size: {}"
                             "".format(num_to_sample, len(self.X)))

        samples = np.random.choice(np.arange(len(self.X)),
                                   size=num_to_sample,
                                   p=self.frequencies,
                                   replace=self.replacement)

        data, words = zip(*[(self.X[x], self.words[x]) for x in samples])
        if isinstance(self.X, np.ndarray):
            return np.asarray(data), words
        return data, words
