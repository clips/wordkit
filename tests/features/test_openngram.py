import pytest
import numpy as np

from wordkit.features import (ConstrainedOpenNGramTransformer,
                              OpenNGramTransformer,
                              WeightedOpenBigramTransformer)


@pytest.fixture()
def ngram_weighted():
    return WeightedOpenBigramTransformer((1.0, .8, .4))


@pytest.fixture
def ngram_nopad():
    return ConstrainedOpenNGramTransformer(n=2, window=2, use_padding=False)


@pytest.fixture
def ngram_pad():
    return ConstrainedOpenNGramTransformer(n=2, window=2, use_padding=True)


@pytest.fixture
def ngram_noconst():
    return OpenNGramTransformer(n=2)


def test_fit_noconst(ngram_noconst):
    ngram_noconst.fit(["dog", "dogs", "doggos"])
    assert len(ngram_noconst.features) == 9


def test_fit_nopad(ngram_nopad):
    ngram_nopad.fit(["dog", "cats", "god"])
    assert len(ngram_nopad.features) == 11


def test_fit_pad(ngram_pad):
    ngram_pad.fit(["dog", "cats", "god"])
    assert len(ngram_pad.features) == 21


def test_transform_nopad(ngram_nopad):
    X = ngram_nopad.fit_transform(["dog", "cats", "god"])
    assert X.shape[0] == 3
    assert X.shape[1] == 11
    assert np.all(X.sum(0) == 1)


def test_ngram(ngram_pad):
    grams = tuple(ngram_pad._ngrams("hoopla"))
    assert grams == tuple([("#", "h"),
                           ("#", "o"),
                           ("h", "o"),
                           ("h", "o"),
                           ("o", "o"),
                           ("o", "p"),
                           ("o", "p"),
                           ("o", "l"),
                           ("p", "l"),
                           ("p", "a"),
                           ("l", "a"),
                           ("l", "#"),
                           ("a", "#")])


def test_ngram_nopad(ngram_nopad):
    grams = tuple(ngram_nopad._ngrams("hoopla"))
    assert grams == tuple([("h", "o"),
                           ("h", "o"),
                           ("o", "o"),
                           ("o", "p"),
                           ("o", "p"),
                           ("o", "l"),
                           ("p", "l"),
                           ("p", "a"),
                           ("l", "a")])


def test_paper_weight(ngram_weighted):
    X = ngram_weighted.fit_transform(["bird"])
    assert np.isclose(X.dot(X.T)[0, 0], 4.44)
