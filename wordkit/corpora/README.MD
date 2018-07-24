# Corpora

Corpus readers transform corpora containing structured information into dictionaries containing the structured information.
These dictionaries can then be sent to other transformers for use.

All readers conform to the following constructor:

```python
Reader(pathtofile, fields, language)
```

where:
* `pathtofile` denotes the path to the file containing the corpus
* `fields` is a tuple denoting which fields to extract
* `language` is the language to extract

All corpus readers use different file formats, so please make sure the file format is correct.

A list of supported corpora can be found [here](../../)

## merge

`merge` is a way to merge two corpora by adding information from one corpus to the other.

It is most commonly used to add frequencies from a corpus of counts to a corpus containing, for example, morphological forms. In this manner, useful corpora with older frequency norms, such as `Celex` can be reused.

```python
from wordkit.corpora import Celex, Subtlex, merge

# Augment celex with subtlex frequencies.
c_words = Celex("path/to/celex/epw.cd").transform()
s_words = Subtlex("path/to/subtlex", language="eng-uk").transform()

# union_fields in the constructor denotes the fields by which the corpora are merged
augmented = merge(from_corpus=s_words,
                  to_corpus=c_words,
                  transfer_fields=("frequency",),
                  merge_fields=("orthography",))

```
