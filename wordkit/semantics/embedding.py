"""Word embeddings."""
# We use reach to load stuff.
from reach import Reach
from ..base import BaseTransformer


class EmbeddingTransformer(BaseTransformer):
    """
    Transforms words into embeddings.

    Word embeddings are pre-trained distributed representations of words.
    The distributed representations are usually obtained by utilizing the
    distributional hypothesis; i.e. that words with the same meaning are
    accompanied by the same words.

    This transformer utilizes a file with stored embeddings.

    Parameters
    ----------
    path_to_embeddings : str
        The path to the embeddings file. The file has to be a space-separated
        file. The header is optional, so this transformer supports either
        Glove or word2vec style headers.

    field : str or None
        The field to use.

    """

    def __init__(self,
                 path_to_embeddings,
                 field=None):
        """Transform words into embeddings."""
        super().__init__(field)
        self.path = path_to_embeddings
        self.features = {}

    def _validate(self, X):
        """Validate the input data."""
        difference = set(X) - self.feature_names
        if difference:
            raise ValueError("Tried to retrieve semantic vectors for words "
                             "for which you do not have embeddings "
                             "".format(difference))

    def fit(self, X):
        """
        Fit the transformer to some data.

        Fitting, in this case, means unpacking the file and loading the
        feature matrix. Only words on which you fit are kept.
        """
        super().fit(X)
        X = self._unpack(X)
        mtr, words = Reach._load(self.path, X)
        self.features = {k: v for k, v in zip(words, mtr)}
        self.vec_len = mtr.shape[1]
        self.feature_names = set(self.features.keys())
        self._is_fit = True
        return self

    def vectorize(self, x):
        """Vectorize."""
        return self.features[x]
