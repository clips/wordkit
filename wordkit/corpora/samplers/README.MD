# Samplers

Samplers are components that can sample from a set of observations according to their frequency.
They can be used to create ecological datasets by putting them after a `wordkit` pipeline.

A sampler takes as input a featurized dataset and, optionally, a list of frequencies.
If no frequencies are passed, the sampler samples uniformly.

A sampler can sample with or without replacement by setting the `replacement` flag to `True` or `False`.
Note that sampling without replacement is only possible if the number of requested items in the call to `sample` is lower than the number of items in the sampler.

We also offer a `BinnedSampler`, which samples while respecting the proportion of items in frequency bins. This is to circumvent the inherent Zipfian distribution in word frequency lists.

## Example

```python
import numpy as np
from wordkit.corpora.samplers import Sampler, BinnedSampler

np.random.seed(44)

words = ["dog", "cat"]
frequencies = [10, 30]

s = Sampler(words, frequencies)
num_to_sample = 6
sampled_words = s.sample(num_to_sample)

# Sample directly from data
X = np.random.randint(0, 2, (100, 10))
frequencies = np.random.randint(0, 1000, 100)

s = Sampler(X, frequencies)
sampled_data = s.sample(10)

b = BinnedSampler(X, frequencies)
binned_data = b.sample(1000)

```
