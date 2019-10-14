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

w = (("dog", "cat", "sheep", "spin"),) * len(featurizers)


@pytest.mark.parametrize("v, words", zip(featurizers, w))
def test_vectorize(v, words):
    X = v.fit_transform(words).astype(np.float)
    m = np.mean(X)
    assert X.shape[0] == len(words)
    assert X.shape[1] == 1024
    X /= np.linalg.norm(X, axis=1)[:, None]
    print(X.dot(X.T))

    assert -.1 < X.dot(X.T)[0, 1] - m < .1
