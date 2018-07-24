# Phonology

This package contains all components related to phonology.
It contains [features](#Feature-sets), [feature extractors](#Feature-Extraction), and [transformers](#Transformers).

## Transformers

These features can either be passed as a `dict` mapping symbols to feature arrays, or as tuples of strings, or tuples of tuples of strings.
As features we expect a tuple of `dict`s, the first containing vowel features, and the second containing consonant features.

In a lot of cases the feature extraction can be postponed by passing a `feature extractor`.
This is usually the easiest method, as it guarantees a parsimonious representation.

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

### WickelFeatureTransformer

The `WickelFeatureTransformer` is an implementation of the featurization procedure introduced by McClelland and Rumelhart in their 1986 paper.
It is intended to produce the same behavior as the `WickelTransformer` above, but has a lower space complexity.

Instead of directly one hot encoding and storing all character trigrams directly, the `WickelFeatureTransformer` assigns a number of binary variables to each character trigram, creating a unique code for each trigram.

This is done by randomly assigning a number of characters to each binary variable. Only if all characters in all positions match will the variable be assigned to this trigram.
This guarantees that trigrams with overlap in characters will be get more similar codes than words without overlap.

The `n` argument behaves as above, and determines the `n`n in character ngrams.
The `num_units` argument determines the number of binary variables, and consequently your output dimensionality.
The `proportion` argument determines the proportion of unique items that are randomly assigned to each unit.
For example, if there are 26 characters in your alphabet, a proportion of .38 will assign 10 characters to each unit for each position.

```python
from wordkit.transformers import WickelFeatureTransformer

words = ["dog", "fog", "hotdog", "colddog"]

w = WickelFeatureTransformer(n=3, num_units=400, proportion=.38)
X = w.fit_transform(words)
```

# Features

This subpackage contains a number of pre-specified features, and is mainly aimed at replicating already existing papers.

# Usage

```python
from wordkit.feature_extraction import PredefinedFeatureExtractor
from wordkit.features import fourteen, dislex_features, plunkett_phonemes
from wordkit.transformers import LinearTransformer, CVTransformer

orth = LinearTransformer(fourteen, field='orthography')

# Use a feature extractor with predefined features.
phon = CVTransformer(PredefinedFeatureExtractor(dislex_features))

# Use a predefined phoneme set.
phon = CVTransformer(plunkett_phonemes)

```

## Features


### Feature extraction

This subpackage contains all the functions and objects involved in feature extraction.
In general, feature extraction denotes the process of extracting features from a set of objects.
Keep in mind that feature extraction is distinct from transformation.
Feature extraction merely determines the set of features, which are then passed to a transformer for further use.

#### Usage

```python
import numpy as np
from wordkit.phonology import PhonemeFeatureExtractor, CVTransformer

p_words = [('k', 'æ', 't'), ('d', 'ɔ', 'ɡ')]
p = PhonemeFeatureExtractor()
v_feats, c_feats = p.extract(p_words)

# These can then be added to a transformer
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

All phonological feature extractors return tuples, where the first item of each tuple is a dictionary of vowel-to-feature mappings, and the second item of each tuple is a dictionary of consonant-to-feature mappings.

#### OneHotPhonemeExtractor

Turns a set of phoneme strings into one hot encoded phonemes.
Each phoneme is assigned an orthogonal vector, and this extractor thus assumes that all phonemes are maximally dissimilar.

#### PhonemeFeatureExtractor

This extractor extracts IPA distinctive features from the input, and assigns each unique value of an IPA feature a one-hot encoded representation.
Note that only features within a feature group are orthogonal.
For example, if there are only three values for a given feature group, the features within that feature group will have a three-bit one-hot encoded vector.

#### PredefinedFeatureExtractor

This extractor first extracts IPA distinctive features, and assigns each of the values in this features a vector, which is predefined.
Predefined phonological features can be found in `wordkit.features.phonology`.
There is no limit on the type of features which can be passed into this extractor, although all values from the same feature group need to have the same dimensionality.

```python
from wordkit.phonology import CVTransformer, PredefinedFeatureExtractor
from wordkit.phonology.features import dislex_features

words = [{"orthography": "dog", "phonology": ('d', 'ɔ', 'ɡ')}]
p = PredefinedFeatureExtractor(dislex_features,
                               field="phonology")
v_feats, c_feats = p.extract(words)
```


### Feature sets

##### binary features

This feature set was defined by ourselves, and is an expansion of the `patpho` feature set. It can encode a wide variety of phonemes, but not all of the phonemes in the International Phonetic Alphabet.
It assigns each of the distinctive features an overlapping binary code, and therefore leads to a more parsimonious representation as the one extracted by the `PhonemeFeatureExtractor`.

##### dislex features

This feature set assigns each single feature a number between `0` and `1`. It can not distinguish between all phonemes in the International Phonetic Alphabet.
Like the orthographic features with the same name, it was defined and used by miikkulainen in the context of the DISLEX model.

### Phoneme sets

Phoneme sets are sets which directly map phoneme characters to features.
We currently include only two of these because of their limited utility; they usually only apply to a subset of available phonemes, and rarely work for the featurization of an entire corpus.

The phoneme sets are all tuples of dictionaries. The first item of each tuple is a mapping from vowels to features, while the second item is a mapping from consonants to features.

#### plunkett_phonemes

These are taken from Plunkett and Marchman (1992):

```
@article{plunkett1993rote,
  title={From rote learning to system building: Acquiring verb morphology in children and connectionist nets},
  author={Plunkett, Kim and Marchman, Virginia},
  journal={Cognition},
  volume={48},
  number={1},
  pages={21--69},
  year={1993},
  publisher={Elsevier}
}
```

#### patpho_binary

Like patpho_real, below, this has been taken from the patpho paper.

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

#### patpho_real

Like patpho_binary above, this feature set has been taken from the patpho paper.
