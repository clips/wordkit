import pytest
import numpy as np

from wordkit.features import (LinearTransformer,
                              OneHotCharacterExtractor,
                              OneHotLinearTransformer,
                              fourteen)


@pytest.fixture
def linear():
    return LinearTransformer(OneHotCharacterExtractor)


@pytest.fixture
def onehotlinear():
    return OneHotLinearTransformer()


@pytest.fixture
def fourteenlinear():
    return LinearTransformer(fourteen)


def test_chars(linear):
    linear.fit(["dog", "cat", "god"])
    assert len(linear.features) == 7
    assert set(linear.features) == {" ", "d", "o", "g", "c", "a", "t"}


def test_features(linear):
    linear.fit(["dog", "cat", "dogdogdog"])
    assert linear.max_word_length == 9
    assert len(linear.features) == 7
    assert linear.vec_len == 7 * 9
    X = linear.transform(["dog", "cat", "dogdogdog"])
    assert X.shape[0] == 3
    assert X.shape[1] == linear.vec_len


def test_inverse_transform(linear):
    w = ["dog", "cat", "dogs", "doggos"]
    X = linear.fit_transform(w)
    w_ = linear.inverse_transform(X)
    assert w == w_


def test_equivalence(linear, onehotlinear):
    w = ["dog", "cat", "dogs", "doggos"]
    X1 = linear.fit_transform(w)
    X2 = onehotlinear.fit_transform(w)

    assert np.all(X1 == X2)


def test_transform_error(linear):
    w = ["dog", "cat"]
    w_test = ["dogi"]

    with pytest.raises(ValueError):
        linear.transform(w)

    linear.fit(w)
    with pytest.raises(ValueError):
        linear.transform(w_test)


def test_transform_error_onehot(onehotlinear):
    w = ["dog", "cat"]
    w_test = ["dogi"]

    with pytest.raises(ValueError):
        onehotlinear.transform(w)

    onehotlinear.fit(w)
    with pytest.raises(ValueError):
        onehotlinear.transform(w_test)


def test_fourteen(fourteenlinear):
    assert len(fourteenlinear.features) == 27


def test_fourteen_features(fourteenlinear):
    w = ["dog", "cat"]
    X = fourteenlinear.fit_transform(w)

    assert X.shape[0] == 2
    assert X.shape[1] == 42
