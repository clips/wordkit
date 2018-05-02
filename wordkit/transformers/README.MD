# Transformers

This is the transformers module in `wordkit`.
It is the heart of `wordkit`, in the sense that most other components of `wordkit`

All transformers in `wordkit` subclass the `sklearn.base.TransformerMixin`, and are therefore outfitted with the expected `fit`, `transform` and `fit_transform` functions.
Because of this, all transformers can be integrated into `sklearn` `Pipeline` or `FeatureUnion` objects.

All transformers take at least a `field` argument, which denotes which `field` from the reader is used to featurize the words.
For all transformers, the following holds: if `field` is set to `None`, the transformer takes as input a list of items, e.g. orthographic. On the other hand, if `field` is set to a value, it takes a list of `dict` as input, where each of the `dict`s in the list _must_ contain the `field` in question.

```python
import numpy as np
from wordkit.transformers import WickelTransformer

# These are equivalent
words = [{"orthography": "dog"}, {"orthography": "cat"}]
words_l = ["dog", "cat"]

# These are equivalent
w = WickelTransformer(n=3, field="orthography")
first = w.fit_transform(words)

w = WickelTransformer(n=3, field=None)
second = w.fit_transform(words_l)

np.all(first == second)

```

## Transformers taking features

Several take as input a set of features during construction.
These features can either be passed as a `dict` mapping symbols to feature arrays.
In the case of phonological transformers, we expect a tuple of `dict`s, the first containing vowel features, and the second containing consonant features.

In a lot of cases the feature extraction can be postponed by passing a _feature_ _extractor_ from [wordkit.feature_extraction](../feature_extraction).
This is usually the easiest method, as it guarantees a parsimonious representation.

### LinearTransformer

The `LinearTransformer` featurizes words by replacing each letter or phoneme in a letter or phoneme string by a vector, and then concatenating the resulting vectors.
It thus assumes that letter encoding is completely position-specific.

```python
from wordkit.transformers import LinearTransformer
from wordkit.feature_extraction import OneHotCharacterExtractor

l = LinearTransformer(OneHotCharacterExtractor, field=None)
X = l.fit_transform(["log", "flog", "fog"])

# Normalized Hamming distance
# Distance between log and flog
dist = (l.max_word_length - X[0].dot(X[1])) / l.max_word_length
# Distance between log and fog
dist_2 = (l.max_word_length - X[0].dot(X[2])) / l.max_word_length
```

### CVTransformer

The `CVTransformer` is based on `patpho`, and uses a Consonant Vowel grid to align consonant and vowel features. This ensures that words with a different number of vowels and consonants in segments are still aligned.

The words `spat` and `pat`, for example, would not be aligned using a purely linear encoding, but are aligned when using a Consonant Vowel grid.

When using the CVTransformer, please consider citing:

```
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
```

As it is a phonological transformer, the CVTransformer expects a tuple of features, one for vowels and another for consonants.

```python
from wordkit.transformers import CVTransformer
from wordkit.feature_extraction import OneHotPhonemeExtractor

p_words = [('p', 'æ', 't'), ('s', 'p', 'æ', 't')]
c = CVTransformer(OneHotPhonemeExtractor, field=None)
X = c.fit_transform(p_words)

c.features
```

### ONCTransformer

The `ONCTransformer` is similar to the `CVTransformer` above, but uses syllable information to group phonemes into Onsets, Nuclei and Codas.
This ensures is more data-intensive, because it requires syllable information whereas the `CVTransformer` only requires phonological information, but is also more accurate.

In contrast to the `CVTransformer`, whose grid must be manually specified, the grid of the `ONCTransformer` is completely data-driven, and determined during fitting.

```python
from wordkit.transformers import ONCTransformer
from wordkit.feature_extraction import OneHotPhonemeExtractor

# Syllables are represented as tuples of tuples.
p_words = [(('p', 'æ', 't'),), (('s', 'p', 'æ', 't'),)]
o = ONCTransformer(OneHotPhonemeExtractor, field=None)
X = o.fit_transform(p_words)

o.features
```
## Transformers without features

This set of transformers generate their own features, and thus do not need predefined features to function.

### OpenNGramTransformer

The `OpenNGramTransformer` featurizes words using open ngrams, which is the set of ordered combinations of ngrams in a word.

Taking bigrams as an example, the `OpenNGramTransformer` turns the word `"salt"` into `{"sa", "sl", "st", "al", "at", "lt"}`. The extracted features are similar to what is known as "character skipgrams".

The main motivation for using the open ngram features is transposition resilience.

If you use the OpenNGramTransformer, please consider citing the following sources:

```
@article{schoonbaert2004letter,
  title={Letter position coding in printed word perception: Effects of repeated and transposed letters},
  author={Schoonbaert, Sofie and Grainger, Jonathan},
  journal={Language and Cognitive Processes},
  volume={19},
  number={3},
  pages={333--367},
  year={2004},
  publisher={Taylor \& Francis}
}

@article{whitney2001brain,
  title={How the brain encodes the order of letters in a printed word: The SERIOL model and selective literature review},
  author={Whitney, Carol},
  journal={Psychonomic Bulletin \& Review},
  volume={8},
  number={2},
  pages={221--243},
  year={2001},
  publisher={Springer}
}
```

The example below shows how `"salt"` and `"slat"` lead to similar encodings.

```python
from wordkit.transformers import OpenNGramTransformer

words = ["salt", "slat"]

o = OpenNGramTransformer(n=2, field=None)
X = o.fit_transform(words)
o.features

# Normalized hamming distance
dist = (X.shape[1] - (X[0].dot(X[1]))) / X.shape[1]
```

### ConstrainedOpenNGramTransformer

The `ConstrainedOpenNGramTransformer` is similar to the `OpenNGramTransformer`, above, with the added constraint that the ngrams can only skip up to a specific number of letters.

If you use this transformer, please cite the sources listed under the `OpenNGramTransformer` heading, above.

```python
from wordkit.transformers import ConstrainedOpenNGramTransformer

words = ["photography", "graphically"]

c = ConstrainedOpenNGramTransformer(n=2, window=2)
c.fit_transform(words)
c.features
```

### WeightedOpenBigramTransformer

The `WeightedOpenBigramTransformer` can only transform bigrams, and assigns each of the bigrams weights depending on the distance between the letters.

If you use this transformer, please cite the sources listed under the `OpenNGramTransformer` heading, above.

```python
from wordkit.transformers import WeightedOpenBigramTransformer

words = ["photography", "graphically"]

# Bigrams with no intervening letters get weight 1,
# bigrams with a single intervening letter get weight .8, and so on.
w = WeightedOpenBigramTransformer(weights=(1., .8, .2))
X = w.fit_transform(words)

```

### WickelTransformer

The `WickelTransformer` turns words into character ngrams.
Every word is padded with `n - 1` dummy characters ("#" by default).

Padding can be turned off by setting `use_padding` to False, but this removes the option of featurizing words which are shorter than `n` characters.

```python
from wordkit.transformers import WickelTransformer

words = ["dog", "fog", "hotdog", "colddog"]

w = WickelTransformer(n=3)
X = w.fit_transform(words)

w_2 = WickelTransformer(n=3, use_padding=False)
X_2 = w_2.fit_transform(words)
```
