"""Unite information from various sources."""
from collections import defaultdict
from copy import deepcopy
from .base import WordStore


def merge(from_corpus,
          to_corpus,
          merge_fields,
          transfer_fields,
          discard=False):
    """
    Augment a corpus by data from another corpus by transfering fields.

    merge joins corpora together by augmenting the information from one corpus,
    the to_corpus, by information from another corpus, the from_corpus.

    For example, given a corpus of orthography - phonology
    combinations and a corpus of frequency norms, merge can be used
    to add frequency norms to the orthography - phonology combinations.

    Merging requires that there is overlap between the fields of the
    corpora. The parameter merge_fields specifies by which fields the corpora
    should be merged.

    Parameters
    ----------
    from_corpus : list of dictionaries
        A list of dictionaries. This corpus will be used to augment
        the information in the to_corpus.
    to_corpus : list of dictionaries
        A list of dictionaries. This corpus will be augmented with the
        information from the from_corpus.
    merge_fields : tuple or string
        The fields to use in determining equivalence between two words in the
        corpora. For example, if merge_fields = ("orthography",), only the
        orthography field is used to compare words. If merge_fields =
        ("orthography", "phonology"), both the information in "orthography" and
        "phonology" needs to be equivalent.
    transfer_fields : tuple or string
        The fields to transfer from the from_corpus to the to_corpus.
    discard : bool
        If this is set to True, words that are not in both corpora are
        discarded.

    Example
    -------
    >>> from wordkit.readers import Celex, Subtlex, merge
    >>> s = Subtlex("path")
    >>> c = Celex("path")
    >>> words_s = s.transform()
    >>> words_c = c.transform()
    >>> new = merge(words_s, words_c, "orthography", "frequency")

    """
    from_keys = set(from_corpus[0].keys())
    to_keys = set(to_corpus[0].keys())

    if isinstance(merge_fields, str):
        merge_fields = (merge_fields,)

    if isinstance(transfer_fields, str):
        transfer_fields = (transfer_fields,)

    all_fields = set(from_keys) & set(to_keys)
    all_diff = set(merge_fields) - all_fields
    if all_diff:
        raise ValueError("{} was passed as merge fields, but these"
                         " fields are not in all your corpora of "
                         " choice.".format(all_diff))

    transfer_fields = set(transfer_fields)
    words_set_from = _hash_words(from_corpus, merge_fields)
    words_set_to = _hash_words(to_corpus, merge_fields)

    keys = set(words_set_from.keys()) and set(words_set_to.keys())
    words_set_from = {k: v for k, v in words_set_from.items() if k in keys}
    words_set_to = {k: v for k, v in words_set_to.items() if k in keys}

    joined_words = deepcopy(to_corpus)
    updated = set()

    for k, indices in words_set_from.items():
        try:
            for idx in words_set_to[k]:
                for from_idx in indices:
                    word = joined_words[idx]
                    information_to_add = from_corpus[from_idx]
                    word.update(information_to_add)
                updated.add(idx)
        except KeyError:
            pass

    if discard:
        joined_words = [x for idx, x in enumerate(joined_words)
                        if idx in updated]

    return WordStore(joined_words)


def _hash_words(words, fields):
    """Hash words by the relevant fields."""
    indices = defaultdict(set)

    for idx, x in enumerate(words):
        indices[tuple([x[field] for field in fields])].add(idx)

    return dict(indices)
