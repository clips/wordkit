"""General functions for use in both phonology and orthography."""
from itertools import chain


def extract_characters(corpus,
                       field,
                       forbidden_characters=()):
    """Extract all characters from a corpus."""
    if isinstance(corpus[0], dict):
        if field == "syllables":
            corpus = [chain.from_iterable(x[field]) for x in corpus]
        else:
            corpus = [x[field] for x in corpus]

    characters = set(chain.from_iterable(corpus))
    characters -= set(forbidden_characters)

    return characters
