# Examples

This folder contains various examples that will allow you to familiarize yourself with wordkit.
Following is a short description of the various examples:

### extracting_words.py

Learn how to extract specific words from your corpora of choice.
Useful for selecting stimuli for psycholinguistic experiments, computational or otherwise.

### featurize_words.py

Learn how to featurize extracted words using a LinearTransformer and a OneHotCharacterExtractor.
These featurized words can then be fed directly to machine learning algorithms.

### improve_celex.py

Learn how to add updated frequency norms to CELEX.
As shown by Brysbaert and New (2009), the CELEX frequency norms are no longer valid, and should be replaced by other frequencies.
In this example, we show how you can supplant the CELEX frequencies by the more accurate SUBTLEX frequencies.

### get_synonyms.py

`wordkit` also supports synonymy through wordnet.
In this example we show how you can retrieve synonyms through wordnet transformers, also crosslingually.
