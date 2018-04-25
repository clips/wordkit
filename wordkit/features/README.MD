# Features

This subpackage contains a number of pre-specified features, and is mainly aimed at replicating already existing papers.
In the case of orthography, a feature set is a dictionary mapping features to values, where values are numpy arrays.

In the case of phonology, a feature set is either a tuple of dictionaries, one mapping vowels to features, and one mapping consontants to features, or a mapping from distinctive features to values. The latter of these two is used in conjunction with a `feature extractor` to create representations of phonemes based on the distinctive features of those phonemes.


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

## Contents

### Orthographical

#### fourteen

This feature set maps all lower-case letters of the alphabet to a set of fourteen binary features. These binary features roughly correspond to the segments found on a microwave display, and therefore encode low-level perceptual similarity between letters.

It was originally defined and used in the Interactive Activation (IA) model.

#### sixteen

This feature set maps all lower-case, upper-case letters, and a whole bunch of other symbols into a set of sixteen binary features. The sixteen-panel encoding is typically found on newer digital watches.

Like the fourteen feature set above, these sixteen features encode low-level perceptual similarity between letters, but also includes a number of symbols, which makes it useful for encoding text which includes punctuation.

#### dislex

This feature sets assigns each character a single number between `0` and `1`, depending on the ratio between black and white pixels in a typical picture of that letter. It was defined by miikkulainen in the context of the DISLEX model, but is also used in other Self-Organizing Map based models.

### Phonological

#### Feature sets

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
