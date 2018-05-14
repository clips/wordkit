# wordkit

![overview](images/wordkit_overview.png)

This is the repository of the `wordkit` package, a `Python 3.X` package for the featurization of words into orthographic and phonological vectors.

A paper that describes `wordkit` was accepted at LREC 2018.
If you use `wordkit` in your research, please cite the following paper:

```
@InProceedings{TULKENS18.249,
  author = {Stephan Tulkens ,Dominiek Sandra and Walter Daelemans},
  title = {WordKit: a Python Package for Orthographic and Phonological Featurization},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {may},
  date = {7-12},
  location = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {979-10-95546-00-9},
  language = {english}
  }
```


Additionally, if you use any of the corpus readers in `wordkit`, you MUST cite the accompanying corpora and transformers.
All of these references can be found in the docstrings of the applicable classes.

`wordkit` makes heavy use of [ipapy](https://github.com/pettarin/ipapy), a package which parses unicode IPA characters into their IPA representations.


## Overview

`wordkit` is a modular system, and contains 5 broad families of components.
The subpackages are extensively documented using separate `README.MD` files.
Feel free to click ahead to find descriptions of the contents of subpackages.

* [Readers](wordkit/readers)
* [Feature extractors](wordkit/feature_extraction)
* [Feature sets](wordkit/features)
* [Transformers](wordkit/transformers)
* [Samplers](wordkit/samplers)

In general, a `wordkit` pipeline consists of one or more readers, which extract structured information from corpora.
This information is then sent to one or more transformers, which are either assigned pre-defined features or a feature extractor.
If a feature extractor is assigned to a transformer, the feature set of that transformer is extracted automatically.

Finally, a sampler can be used to sample from the resulting items.

## Corpora

`wordkit` currently offers readers for the following corpora.

#### BPAL

[Download](http://www.pc.rhul.ac.uk/staff/c.davis/Utilities/B-Pal.zip)

You have to extract the `nwphono.txt` file from the `.exe` file.
The corpus is not for download in a more practical fashion.

[Publication](http://www.pc.rhul.ac.uk/staff/c.davis/Articles/Davis_Perea_in_press.pdf)

```
Fields:     Orthography, Phonology, Frequency
Languages:  Spanish
```

#### Celex

Currently not freely available.  

```
Fields:     Orthography, Phonology, Syllables, Frequency
Languages:  Dutch, German, English
```

**WARNING:** the Celex frequency norms are no longer thought to be correct. Please use the `SUBTLEX` frequencies instead.
You can use the Celex corpus with `SUBTLEX` frequency norms by using a [`CorpusAugmenter`](wordkit/readers#CorpusAugmenter).   
If you use `CELEX` frequency norms at a psycholinguistic conference, you _will_ get yelled at.

#### CMUDICT

[Download](https://github.com/cmusphinx/cmudict)

We can read the `cmudict.dict` file from the above repository.

```
Fields:     Orthography, Syllables  
Languages:  American English
```

#### Deri

[Download](https://drive.google.com/open?id=0B7R_gATfZJ2aUWsxbF9CSEo1Y00)

Download the `pron_data.tar.gz` file, and unzip it. We use the `gold_data_train` file.

[Publication](http://isi.edu/~aderi/papers/g2p.pdf)

```
Fields:     Orthography, Phonology  
Languages:  lots
```

**WARNING:** we manually checked the Dutch, Spanish and German phonologies in this corpus, and a lot of them seem to be incorrectly transcribed or extracted. Only use this corpus if you don't have another resource for your language.

#### Lexique

[Download](http://lexique.org/telLexique.php)  

Download the zip file, we use the `lexique382.txt` file.

[Publication](https://link.springer.com/content/pdf/10.3758/BF03195598.pdf)

Note that this is the publication for Lexique version 2. Lexique 3 does not seem to have an associated publication in English.

```
Fields:     Orthography, Phonology, Frequency, Syllables  
Languages:  French
```

**NOTE:** the currently implemented reader is for version 3.82 (the most recent version as of May 2018) of Lexique.

#### SUBTLEX

Check the link below for the various SUBTLEX corpora and their associated publications.
We support all of the formats from the link below.

[Link](http://crr.ugent.be/programs-data/subtitle-frequencies)

```
Fields:     Orthography, Frequency  
Languages:  Dutch, American English, Greek
            British English, Polish, Chinese,
            Spanish
```

## Example

```python
from wordkit.readers import Celex
from wordkit.transformers import LinearTransformer, WickelTransformer
from wordkit.features import fourteen
from sklearn.pipeline import FeatureUnion

# The fields we want to extract from our corpora.
fields = ('orthography', 'frequency', 'phonology', 'syllables')

# Filter function
# A filter function can be added to a corpus to filter out any
# words which do not conform to some criterion, e.g. frequency
# constraints.
# In this case, we use it to remove punctuation marks for which
# we do not have any features.
# We also use it to select only monosyllables.
fil = lambda x: not set(x['orthography']).intersection({"'", ',', '-', '/', '.'}) and len(x['syllables']) == 1

# We set merge duplicates to True because some of
# the distinctions in Celex (e.g. POS) do not matter to us.

# Link to epl.cd
english = Celex("epl.cd",
                fields=fields,
                merge_duplicates=True)

# Link to dpl.cd
dutch = Celex("dpl.cd",
              fields=fields,
              language='nld',
              merge_duplicates=True)

words = english.transform(filter_function=fil)
words.extend(dutch.transform(filter_function=fil))

# words[0] =>
# {'frequency': 1267035,
# 'orthography': 'a',
# 'phonology': ('e', 'ɪ'),
# 'syllables': (('e', 'ɪ'),)}

# You can also query specific words
wind = english.transform(["wind"]) + dutch.transform(["wind"])

# This gives
# wind =>
# [{'frequency': 16.154509673599474,
#   'orthography': 'wind',
#   'phonology': ('w', 'a', 'ɪ', 'n', 'd'),
#   'syllables': (('w', 'a', 'ɪ', 'n', 'd'),)},
#  {'frequency': 116.57008309321614,
#   'orthography': 'wind',
#   'phonology': ('w', 'ɪ', 'n', 'd'),
#   'syllables': (('w', 'ɪ', 'n', 'd'),)},
#  {'frequency': 116.49571380983738,
#   'orthography': 'wind',
#   'phonology': ('w', 'ɪ', 'n', 't'),
#   'syllables': (('w', 'ɪ', 'n', 't'),)}]
# Note that the frequency has been divided scaled to 1M words.

# Now, let's transform into features
# Orthography is a linear transformer with the fourteen segment feature set.
o = LinearTransformer(fourteen, field='orthography')
# For phonology we use Wickelphones.
p = WickelTransformer(n=3, field='phonology')

featurizer = FeatureUnion([("o", o), ("p", p)])

# Fit and transform the featurizers.
X = featurizer.fit_transform(words)
# A (7650, 4877) matrix.

# Get the feature vector length for each featurizer
o.vec_len # 112
p.vec_len # 4765
```

## Experiments

The code for replicating the experiments in the `wordkit` paper can be found [here](https://github.com/stephantul/lrec2018)

## Requirements

- sklearn
- ipapy
- numpy
- regex
- pandas (for some of the corpus readers)


## Contributors

Stéphan Tulkens

## License

GPL v3
