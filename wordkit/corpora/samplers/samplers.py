"""Sampler class."""
import numpy as np
from sklearn.base import TransformerMixin
from collections import defaultdict


class Sampler(TransformerMixin):
    """
    Sample from a list of words, based on the frequency of those words.

    Parameters
    ----------
    X : numpy array or list of items
        Your vectorized data.
    frequencies : numpy array or list of floats, default None.
        The frequencies of your input data. If this is None, the Sampler
        will sample uniformly.
    replacement : bool
        Whether to sample with or without replacement.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(44)
    >>> from wordkit.samplers import Sampler
    >>> words = ["dog", "cat"]
    >>> frequencies = [10, 30]
    >>> s = Sampler(words, frequencies)
    >>> num_to_sample = 6
    >>> sampled_data = s.sample(num_to_sample)
    >>> sampled_data
    array(['cat', 'dog', 'cat', 'cat', 'cat', 'cat'], dtype='<U3')

    """

    def __init__(self,
                 X,
                 frequencies=None,
                 replacement=True):
        """Sample from a distribution over words."""
        if frequencies is None:
            frequencies = np.ones(len(X))
        else:
            frequencies = np.asarray(frequencies)
        assert(len(frequencies) == len(X))

        self.X = X
        self.frequencies = frequencies / np.sum(frequencies)
        self.replacement = replacement

    def sample(self, num_to_sample):
        """
        Sample from the data.

        Parameters
        ----------
        num_to_sample : int
            The number of items to sample.

        Returns
        -------
        features : np.array
            A matrix of sampled data.

        """
        if not self.replacement and num_to_sample > len(self.X):
            raise ValueError("You tried to sample without replacement from "
                             "a set which is smaller than your sample size "
                             ": sample size: {} set size: {}"
                             "".format(num_to_sample, len(self.X)))

        samples = np.random.choice(np.arange(len(self.X)),
                                   size=num_to_sample,
                                   p=self.frequencies,
                                   replace=self.replacement)

        # We can't use smart indexing because we don't know whether our
        # base data is an array.
        return np.take(self.X, samples, axis=0)


class BinnedSampler(object):
    """
    A sampler that respects the proportions in your population.

    A conventional Sampler samples all items independently according to their
    frequency. Therefore, if you sample many smaller samples from your
    distribution, many of these samples will share some high-frequency words,
    while other words will never occur in any of the samples.

    One of the ways to circumvent this is to constrain our sampling regime
    to sample proportionally from a set of frequency bins. Samples using
    this sampling scheme will have the same frequency distribution as your
    population.

    It works by binning the data according to their frequency, and noting the
    proportion of each bin to the total frequency of the sample.
    Each sample will randomly consist of the same proportion of items from
    each bin.

    Parameters
    ----------
    X : numpy array
        Your vectorized data.
    frequencies : numpy array or list of floats
        The frequencies of your data. This can not be None.
    bin_width : float
        The bin width
    replacement : bool
        Whether the same item can occur more than once in a single sample.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(44)
    >>> from wordkit.samplers import Sampler
    >>> words = ["dog", "cat"]
    >>> frequencies = [10, 30]
    >>> s = BinnedSampler(words, frequencies)
    >>> num_to_sample = 6
    >>> sampled_data = s.sample(num_to_sample)
    >>> sampled_data
    array(['cat', 'dog', 'cat', 'cat', 'cat', 'cat'], dtype='<U3')

    """

    def __init__(self, X, frequencies, bin_width=1.0, replacement=False):
        """Initialize the sampler."""
        self.total = len(X)
        self.bin_width = bin_width
        self.replacement = replacement
        w = defaultdict(list)
        frequencies = np.asarray(frequencies) // bin_width
        for x, freq in zip(X, frequencies):
            w[int(freq)].append(x)
        self.bins = [np.array(v) for k, v in sorted(w.items())]
        self.proportions = np.array([len(x) / self.total for x in self.bins])

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

        """
        if num_to_sample > self.total and not self.replacement:
            raise ValueError("You requested a bigger sample than your "
                             "population, but replacement was set to False. "
                             "Please set replacement to True, or lower your "
                             "sample size.")
        result = []
        props = np.floor(self.proportions * num_to_sample).astype(np.int32)
        error = num_to_sample - props.sum()
        for x in np.random.randint(0, len(props), error):
            props[x] += 1
        for idx, x in enumerate(props):
            if x == 0:
                continue
            bin = self.bins[idx]
            idxes = np.random.choice(len(bin), x, replace=self.replacement)
            result.extend(np.take(bin, idxes, axis=0))

        return np.asarray(result)
