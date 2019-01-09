"""Extract words from a corpus, and filter them using anonymous functions."""
from wordkit.corpora import Subtlex


if __name__ == "__main__":

    # Subtlex is a specific corpus reader.
    # Each corpus reader is associated with a certain corpus, or type of corpus
    # fields denotes which type of information is retrieved from the corpus.
    # In this case, we say we want to extract orthography and frequency from
    # the corpus.

    # By doing this, any words extracted from the corpus will be guaranteed
    # to have the following fields.

    # NOTE: replace path_to_subtlex with the path to subtlex on your machine.
    corpus = Subtlex("path_to_subtlex",
                     fields=("orthography", "frequency"))

    # Now we will extract specific words by adding filters.
    # by not passing any arguments, we simply extract all words.
    words = corpus.transform()

    # words is a list of dictionaries.
    print("{} words in corpus".format(len(words)))
    print("First 10 words:")
    print(words[:10])

    # By passing filters, we can limit the words we get to specific ones.
    # The syntax of the filter is as follows:
    # field_name=function
    # This function is then applied to each item in the corpus.
    # Only if all filters evaluate to True will the item be selected.

    # This filter only extracts words whose orthographic form starts
    # with an 'a'.

    # Note the use of the lambda, which is an anonymous function.
    words = corpus.transform(orthography=lambda x: x[0] == 'a')
    # You can also define other functions.
    # this one is equivalent to the lambda above.

    def filter_function(x):
        """Filter words with an 'a'."""
        if x[0] == 'a':
            return True
        return False

    words = corpus.transform(orthography=filter_function)

    # First 10 words starting with an a
    print("Words left after filtering: {}".format(len(words)))
    print("First 10 words after filtering:")
    print(words[:10])

    # After filtering, we can filter again.
    # Let's filter on frequency by throwing away any words with frequencies
    # smaller than 10.
    # Note that in this case we filter again.
    words = words.filter(frequency=lambda x: x >= 10)

    print("Words left after filtering: {}".format(len(words)))
    print("First 10 words after filtering:")
    print(words[:10])
