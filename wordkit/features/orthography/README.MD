# Orthographic features

This package contains all the orthographical parts of wordkit.
It contains [features](#feature-sets), [feature extractors](#feature-extraction), and [transformers](#transformers).

## Transformers

### LinearTransformer

The `LinearTransformer` featurizes words by replacing each letter or phoneme in a letter or phoneme string by a vector, and then concatenating the resulting vectors.
It thus assumes that letter encoding is completely position-specific.

```python
from wordkit.features import LinearTransformer, OneHotCharacterExtractor

l = LinearTransformer(OneHotCharacterExtractor, field=None)
X = l.fit_transform(["log", "flog", "fog"])

# Normalized Hamming distance
# Distance between log and flog
dist = (l.max_word_length - X[0].dot(X[1])) / l.max_word_length
# Distance between log and fog
dist_2 = (l.max_word_length - X[0].dot(X[2])) / l.max_word_length
```

### OpenNGramTransformer

The `OpenNGramTransformer` featurizes words using open ngrams, which is the set of ordered combinations of ngrams in a word.

Taking bigrams as an example, the `OpenNGramTransformer` turns the word `"salt"` into `{"sa", "sl", "st", "al", "at", "lt"}`. The extracted features are similar to what is known as "character skipgrams".

The main motivation for using the open ngram features is that words with deleted or transposed letters still get represented correctly.

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
from wordkit.features import OpenNGramTransformer

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
from wordkit.features import ConstrainedOpenNGramTransformer

words = ["photography", "graphically"]

c = ConstrainedOpenNGramTransformer(n=2, window=2)
c.fit_transform(words)
c.features
```

### WeightedOpenBigramTransformer

The `WeightedOpenBigramTransformer` can only transform bigrams, and assigns each of the bigrams weights depending on the distance between the letters.

If you use this transformer, please cite the sources listed under the `OpenNGramTransformer` heading, above.

```python
from wordkit.features import WeightedOpenBigramTransformer

words = ["photography", "graphically"]

# Bigrams with no intervening letters get weight 1,
# bigrams with a single intervening letter get weight .8, and so on.
w = WeightedOpenBigramTransformer(weights=(1., .8, .2))
X = w.fit_transform(words)

```

### NGramTransformer

The `NGramTransformer` turns words into character ngrams.
Every word is padded with `n - 1` dummy characters ("#" by default).

Padding can be turned off by setting `use_padding` to False, but this removes the option of featurizing words which are shorter than `n` characters.

```python
from wordkit.features import NGramTransformer

words = ["dog", "fog", "hotdog", "colddog"]

w = NGramTransformer(n=3)
X = w.fit_transform(words)

w_2 = NGramTransformer(n=3, use_padding=False)
X_2 = w_2.fit_transform(words)
```

# Feature extraction

This subpackage contains all the functions and objects involved in feature extraction.
In general, feature extraction denotes the process of extracting features from a set of objects.
Keep in mind that feature extraction is distinct from transformation.
Feature extraction merely determines the set of features, which are then passed to a transformer for further use.

## Usage

```python
import numpy as np
from wordkit.features import PhonemeFeatureExtractor, OneHotCharacterExtractor

o_words = ["dog", "cat"]
o = OneHotCharacterExtractor()
o_feats = o.extract(o_words)

p_words = [('k', 'æ', 't'), ('d', 'ɔ', 'ɡ')]
p = PhonemeFeatureExtractor()
v_feats, c_feats = p.extract(p_words)

# These can then be added to a transformer
from wordkit.features import CVTransformer
c = CVTransformer((v_feats, c_feats))
transformed = c.fit_transform(p_words)

# Feature extractors can be directly added to transformers.
c = CVTransformer(PhonemeFeatureExtractor)
transformed_2 = c.fit_transform(p_words)

# Both methods are equivalent
assert np.all(transformed == transformed_2)

# using a dictionary
words = [{"orthography": "cat", "phonology": ('k', 'æ', 't')},
         {"orthography": "dog", "phonology": ('d', 'ɔ', 'ɡ')}]

# field must be set because we use a dictionary
c = CVTransformer(PhonemeFeatureExtractor, field="phonology")
transformed_3 = c.fit_transform(words)
```

## Features

To use the `LinearTransformer`, words need to be turned into features.
This can be done by using a feature extractor, or by using predefined feature sets.
Both can be directly passed into the transformer.

### Feature extraction

In general, feature extraction denotes the process of extracting features from a set of objects.
Keep in mind that feature extraction is distinct from transformation.
Feature extraction merely determines the set of features, which are then passed to a transformer for further use.

```python
import numpy as np
from wordkit.features import LinearTransformer, OneHotCharacterExtractor

o_words = ["dog", "cat"]
o = OneHotCharacterExtractor()
o_feats = o.extract(o_words)

# Both of these are the same.
l = LinearTransformer(o_feats)
l_2 = LinearTransformer(OneHotCharacterExtractor)

l.fit_transform(o_words) == l_2.fit_transform(o_words)

```

#### OneHotCharacterExtractor

Turns a set of strings into one hot encoded character features.
This is the simplest method, and assumes zero similarity between individual characters.

### Feature sets

We also offer a number of pre-specified features, which are mainly aimed at replicating already existing papers.
These can directly be passed in to the `LinearTransformer`.

```python
from wordkit.features import fourteen, sixteen

l = LinearTransformer(fourteen)
```

#### fourteen

This feature set maps all lower-case letters of the alphabet to a set of fourteen binary features. These binary features roughly correspond to the segments found on a microwave display, and therefore encode low-level perceptual similarity between letters.

It was originally defined and used in the Interactive Activation (IA) model.

#### sixteen

This feature set maps all lower-case, upper-case letters, and a whole bunch of other symbols into a set of sixteen binary features. The sixteen-panel encoding is typically found on newer digital watches.

Like the fourteen feature set above, these sixteen features encode low-level perceptual similarity between letters, but also includes a number of symbols, which makes it useful for encoding text which includes punctuation.

#### dislex

This feature sets assigns each character a single number between `0` and `1`, depending on the ratio between black and white pixels in a typical picture of that letter. It was defined by miikkulainen in the context of the DISLEX model, but is also used in other Self-Organizing Map based models.
