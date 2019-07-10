# Corpora

Corpus readers transform corpora containing structured information into dictionaries containing the structured information.
These dictionaries can then be sent to other transformers for use.

All readers conform to the following function specification:

```python
reader(pathtofile, fields)
```

where:
* `pathtofile` denotes the path to the file containing the corpus
* `fields` is a tuple denoting which fields to extract

All corpus readers use different file formats, so please make sure the file format is correct.

A list of supported corpora can be found [here](../../)

All corpora function return a [Frame](frame)

```python
from wordkit.corpora import subtlex
w = subtlex("path", fields=("frequency", "orthography"))
```

## Adding your own readers

Generic corpora (either in csv or xsl(x) format) can be read using the `reader` function.

```python
# Assume some CSV
from wordkit.corpora import reader
w = reader("path_to_weird_csv", fields=("my_frequency_1", "wordness12"))
# w is now a frame.
```

## Frame

The most important part of the corpus readers in `wordkit` is the `Frame`.
A `Frame` is a pandas-like dataframe that is a bit more restricted, and therefore a bit easier to use.
