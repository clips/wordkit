"""Unite information from various sources."""
from collections import defaultdict
from copy import deepcopy


class CorpusAugmenter(object):
    """
    Augments a corpus by data from another corpus by transfering fields.

    A CorpusAugmenter joins corpora together by augmenting the information
    from one corpus, the to_corpus, by information from another corpus, the
    from_corpus.

    For example, given a corpus of orthography - phonology
    combinations and a corpus of frequency norms, a CorpusAugmenter can be used
    to add frequency norms to the orthography - phonology combinations.

    A CorpusAugmenter requires that there is overlap between the fields of the
    corpora. The parameter union_fields specifies by which fields the corpora
    should be compared.

    Parameters
    ----------
    from_corpus : instance of a Reader
        An initialized Wordkit reader. This corpus will be used to augment
        the information in the to_corpus.
    to_corpus : instance of a Reader
        An initialized Wordkit reader. This corpus will be augmented with the
        information from the from_corpus.
    transfer_fields : tuple, default ()
        The fields to transfer from one corpus to the other. All the transfer
        fields must be in the from_corpus. If the to corpus has information in
        the transfer fields, it will be overwritten.
    union_fields : tuple, default None
        The fields to use in determining equivalence between two words in the
        corpora. For example, if union_fields = ("orthography",), only the
        orthography field is used to compare words. If union_fields =
        ("orthography", "phonology"), both the information in "orthography" and
        "phonology" needs to be equivalent.

    """

    def __init__(self,
                 from_corpus,
                 to_corpus,
                 transfer_fields=(),
                 union_fields=None):
        """Initialize the augmenter."""
        if union_fields is not None:

            all_fields = set(from_corpus.fields) & (set(to_corpus.fields))
            all_diff = set(union_fields) - all_fields
            if all_diff:
                raise ValueError("{} was passed as union fields, but these"
                                 " fields are not in all your corpora of "
                                 " choice.".format(all_diff))
        else:
            union_fields = set(from_corpus.fields) & set(to_corpus.fields)

        diff_transfer = set(transfer_fields) - set(from_corpus.fields)
        if diff_transfer:
            raise ValueError("{} was passed as transfer fields, but these "
                             "fields are not all in your corpora of choice."
                             "".format(diff_transfer))

        self.union_fields = set(union_fields)
        self.transfer_fields = set(transfer_fields)
        self.from_corpus = from_corpus
        # modify from corpus fields
        # pretty dangerous, because this is a permanent change.
        fields_to_keep = self.union_fields | self.transfer_fields
        self.from_corpus.fields = {k: v
                                   for k, v in self.from_corpus.fields.items()
                                   if k in fields_to_keep}
        self.to_corpus = to_corpus

    def transform(self, words=()):
        """Transform a set of words by augmenting the corpora."""
        words_from = self.from_corpus.transform(words)
        words_set_from = self._hash_words(words_from)
        # Speed up: we only need to look for words that were retrieved from
        # the from corpus.
        if "orthography" in self.union_fields:
            words = [x['orthography'] for x in words_from]

        words_to = self.to_corpus.transform(words)
        word_set_to = self._hash_words(words_to)

        joined_words = []

        for k, indices in words_set_from.items():
            try:
                for idx in word_set_to[k]:
                    for from_idx in indices:
                        word = deepcopy(words_to[idx])
                        information_to_add = words_from[from_idx]
                        word.update(information_to_add)
                        joined_words.append(word)
            except KeyError:
                pass

        return joined_words

    def _hash_words(self, words):
        """Hash words by the relevant fields."""
        indices = defaultdict(set)

        for idx, x in enumerate(words):
            indices[tuple([x[field] for field in self.union_fields])].add(idx)

        return dict(indices)
