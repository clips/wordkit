import pytest
import numpy as np


from wordkit.features import (KanervaLinearTransformer,
                              KanervaNGramTransformer,
                              KanervaOpenNGramTransformer,
                              KanervaConstrainedOpenNGramTransformer,
                              PlateLinearTransformer,
                              PlateNGramTransformer,
                              PlateOpenNGramTransformer,
                              PlateConstrainedOpenNGramTransformer)

featurizers = (KanervaLinearTransformer(1024),
               KanervaNGramTransformer(1024,
                                       n=3,
                                       use_padding=False),
               KanervaOpenNGramTransformer(1024, n=2),
               KanervaConstrainedOpenNGramTransformer(1024,
                                                      n=2,
                                                      window=2,
                                                      use_padding=False),
               PlateLinearTransformer(1024),
               PlateNGramTransformer(1024,
                                     n=3,
                                     use_padding=False),
               PlateOpenNGramTransformer(1024, n=2),
               PlateConstrainedOpenNGramTransformer(1024,
                                                    n=2,
                                                    window=2,
                                                    use_padding=False))

w = (("dog", "cat", "ruz", "spin"),) * len(featurizers)


@pytest.mark.parametrize("v, words", zip(featurizers, w))
def test_vectorize(v, words):
    X = v.fit_transform(words).astype(np.float)

    assert X.shape[0] == len(words)
    assert X.shape[1] == 1024
    X /= np.linalg.norm(X, axis=1)[:, None]
    print(X.dot(X.T))

    if v._dtype == np.bool:
        l, u = .3, .7
    else:
        l, u = -.2, .2

    x_ = X.dot(X.T)
    mask = ~np.eye(x_.shape[0], dtype=np.bool)
    assert np.all(l < x_[mask]) & np.all(x_[mask] < u)
