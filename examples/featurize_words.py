"""Extract words from a corpus, and featurize them."""
from wordkit.corpora import Subtlex
from wordkit.orthography import LinearTransformer, OneHotCharacterExtractor


if __name__ == "__main__":

    # We instantiate a Subtlex corpus reader.
    # NOTE: replace path_to_subtlex by the path to subtlex on your own file.
    corpus = Subtlex("path_to_subtlex", fields=("orthography", "frequency"))

    # We limit ourselves to words of length 4.
    # We again use a lambda function that returns a boolean value.
    # only words for which this lambda function evaluates to True get
    # retrieved.
    words = corpus.transform(orthography=lambda x: len(x) == 4)

    print("The first 10 words after filtering:")
    print(words[:10])

    # We initialize the LinearTransformer, which puts features in slots by
    # concatenating them.
    # Note the field argument. This specifies on which field the transformer
    # operates. In this case, we would like to operate on orthographic
    # information.

    # The OneHotCharacterExtractor is feature set, which specifies which
    # features are extracted from the orthographic string.
    # As the name implies, in this case it extracts one hot encoded characters.
    lin = LinearTransformer(OneHotCharacterExtractor, field="orthography")

    # We then need to fit the transformer to the data.
    # We also immediately transform our data using the fit_transform function.
    X = lin.fit_transform(words)

    # X is a matrix with a number of rows equal to the number of words,
    # and a number of columns equal to the number of features our transformer
    # created.
    print("The shape of the resulting matrix: {}".format(X.shape))
