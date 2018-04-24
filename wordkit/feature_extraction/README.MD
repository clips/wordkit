# feature extraction

This subpackage contains all the functions and objects involved in feature extraction.
In general, feature extraction denotes the process of extracting features from a set of objects.
Keep in mind that feature extraction is distinct from transformation.
Feature extraction merely determines the set of features, which are then passed to a transformer for further use.

## Usage

```python
import numpy as np
from wordkit.feature_extraction import PhonemeFeatureExtractor, OneHotCharacterExtractor

o_words = ["dog", "cat"]
o = OneHotCharacterExtractor()
o_feats = o.extract(o_words)

p_words = [('k', 'æ', 't'), ('d', 'ɔ', 'ɡ')]
p = PhonemeFeatureExtractor()
v_feats, c_feats = p.extract(p_words)

# These can then be added to a transformer
from wordkit.transformers import CVTransformer
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

## Contents

All feature extractors take as input a list of dictionaries, or a list of items.
This depends on whether the `field` parameter of the feature extractor is set.

### orthography

#### OneHotCharacterExtractor

Turns a set of strings into one hot encoded character features.
This is the most simple method, and assumes zero similarity between individual characters.
If you are using one

### phonology

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
from wordkit.feature_extraction import PredefinedFeatureExtractor
from wordkit.features import miikkulainen_features
from wordkit.transformers import CVTransformer

words = [{"orthography": "dog", "phonology": ('d', 'ɔ', 'ɡ')}]
p = PredefinedFeatureExtractor(miikkulainen_features,
                               field="phonology")
v_feats, c_feats = p.extract(words)
```
