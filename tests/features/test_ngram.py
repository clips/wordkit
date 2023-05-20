import pytest
import numpy as np

from wordkit.features import NGramTransformer


@pytest.fixture
def ngram_nopad():
    return NGramTransformer(n=3, use_padding=False)


@pytest.fixture
def ngram_pad():
    return NGramTransformer(n=3, use_padding=True)


def test_fit_nopad(ngram_nopad):
    ngram_nopad.fit(["dog", "cats", "god"])
    assert len(ngram_nopad.features) == 4


def test_fit_pad(ngram_pad):
    ngram_pad.fit(["dog", "cats", "god"])
    assert len(ngram_pad.features) == 16


def test_transform_nopad(ngram_nopad):
    X = ngram_nopad.fit_transform(["dog", "cats", "god"])
    assert X.shape[0] == 3
    assert X.shape[1] == 4
    assert np.all(X.sum(0) == 1)


def test_transform_pad(ngram_pad):
    X = ngram_pad.fit_transform(["dog", "cats", "god"])
    assert X.shape[0] == 3
    assert X.shape[1] == 16
    assert np.all(X.sum(0) == 1)


def test_ngram(ngram_nopad):
    grams = tuple(ngram_nopad._ngrams("hoopla"))
    assert grams == tuple([("h", "o", "o"),
                           ("o", "o", "p"),
                           ("o", "p", "l"),
                           ("p", "l", "a")])


def test_ngram_pad(ngram_pad):
    grams = tuple(ngram_pad._ngrams("hoopla"))
    assert grams == tuple([("#", "#", "h"),
                           ("#", "h", "o"),
                           ("h", "o", "o"),
                           ("o", "o", "p"),
                           ("o", "p", "l"),
                           ("p", "l", "a"),
                           ("l", "a", "#"),
                           ("a", "#", "#")])


def test_decompose(ngram_nopad):
    grams1 = tuple(ngram_nopad._ngrams("hoopla"))
    weights, grams2 = zip(*ngram_nopad._decompose("hoopla"))
    assert grams1 == grams2
    assert all([w == 1.0 for w in weights])


def test_decompose_pad(ngram_pad):
    grams1 = tuple(ngram_pad._ngrams("hoopla"))
    weights, grams2 = zip(*ngram_pad._decompose("hoopla"))
    assert grams1 == grams2
    assert all([w == 1.0 for w in weights])
