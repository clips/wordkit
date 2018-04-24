# Readers

Readers transform corpora containing structured information into a dictionaries containing the structured information.
These dictionaries can then be sent to other transformers for use.

All readers conform to the following constructor:

```python
Reader(pathtofile, fields, language)
```

where:
* `pathtofile` denotes the path to the file containing the corpus
* `fields` is a tuple denoting which fields to extract
* `language` is the language to extract

All corpora readers use different file formats, so please make sure the file format is correct.

A list of supported corpora can be found [here](../../)

## CorpusAugmenter

The `CorpusAugmenter` is a way to merge two corpora by adding information from one corpus to the other.

It is most commonly used to add frequencies from a corpus of counts to a corpus containing, for example, morphological forms. In this manner, useful corpora with older frequency norms, such as `Celex` can be reused.

```python
from wordkit.readers import Celex, Subtlex, CorpusAugmenter

# Augment celex with subtlex frequencies.
c = Celex("path/to/celex/epw.cd")
s = Subtlex("path/to/subtlex", language="eng-uk")

# union_fields in the constructor denotes the fields by which the corpora are merged
augmenter = CorpusAugmenter(from_corpus=s, to_corpus=c, union_fields=("orthography,"))

new_corpus = augmenter.transform()
```
