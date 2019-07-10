# Semantics

This module contains the semantics parts of the wordkit package.
It currently contains several transformers for use with semantic data.

Note that we currently only supply semantic data in the form of WordNet data (through the Wordnet Reader).
We also support word embeddings through [reach](www.github.com/stephantul/reach)

## Transformers

### OneHotSemanticsTransformer

The OneHotSemanticsTransformer transforms synsets from WordNet into one binary values that indicate the presence or absence of that synset.
This method, although very coarse, still allows you to capture broad semantic structure, as lemmas which share synsets should be related.

Because synsets which are not shared between words do not contribute anything meaningful they can be pruned using the `prune` switch.

Choosing to set the `prune` switch to True also skips the validation step during input, so be sure to only pass in synsets which you know are in your dataset.

```python
from wordkit.corpora import WordNet
from wordkit.semantics import OneHotSemanticsTransformer

w = WordNet("path/to/wordnet.tsv")
words = w.transform()

o = OneHotSemanticsTransformer()
X = o.fit_transform(o)
```


### HypernymSemanticsTransformer

Like the OneHotSemanticsTransformer, above, the HypernymSemanticsTransformer uses synset information from WordNet to create semantic vectors.
Unlike the OneHotSemanticsTransformer, it does this by emitting all hypernyms and their meronyms for a synset input.
It therefore leads to different semantic relations compared to the OneHotSemanticsTransformer.

If you use this transformer, please cite:

```
@techreport{harm2002building,
  title={Building large scale distributed semantic
         feature sets with WordNet},
  author={Harm, MW},
  year={2002},
  institution={Technical Report PDP-CNS-02-1, Carnegie Mellon University}
}
```

This Transformer takes an optional `prune` switch, which prunes synsets that occur only once.
It also take a `part_meronyms` switch, which denotes whether meronyms should be extracted in addition to hypernyms.

```python
from wordkit.corpora import WordNet
from wordkit.semantics import HypernymSemanticsTransformer

w = WordNet("path/to/wordnet.tsv")
words = w.transform()

o = HypernymSemanticsTransformer()
X = o.fit_transform(o)
```


### EmbeddingTransformer

This transformer translates orthographic words into their semantic vectors.
The semantic vectors have to be supplied by putting them into a word2vec-style file, which is a csv-file that uses a space as a separator.
The embeddings are loaded using `reach`.
Any embeddings not present in the fitting stage are discarded, so make sure you know beforehand which words you want to keep.

```python
from wordkit.semantics import EmbeddingTransformer

e = EmbeddingTransformer("path/to/my/embeddings.vec")
X = e.fit_transform(["cat", "dog"])

print(X)
```
