"""Put things on a grid."""
from wordkit.features.phonology.cv import CVTransformer
from wordkit.features.phonology.feature_extraction import OneHotPhonemeExtractor


def put_on_grid(words, field="phonology", left=True):
    """Puts phonemes on a grid."""
    if field is not None:
        words = [x[field] for x in words]
    c = CVTransformer(OneHotPhonemeExtractor, left=left)
    c.fit(words)
    return [c.put_on_grid(x) for x in words]
