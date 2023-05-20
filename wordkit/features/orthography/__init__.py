"""Orthography."""
from wordkit.features.orthography.feature_extraction import IndexCharacterExtractor, OneHotCharacterExtractor
from wordkit.features.orthography.features import dislex, fourteen, sixteen
from wordkit.features.orthography.linear import LinearTransformer, OneHotLinearTransformer
from wordkit.features.orthography.ngram import NGramTransformer
from wordkit.features.orthography.openngram import (
    ConstrainedOpenNGramTransformer,
    OpenNGramTransformer,
    WeightedOpenBigramTransformer,
)

__all__ = [
    "OneHotCharacterExtractor",
    "LinearTransformer",
    "OpenNGramTransformer",
    "ConstrainedOpenNGramTransformer",
    "WeightedOpenBigramTransformer",
    "NGramTransformer",
    "OneHotLinearTransformer",
    "IndexCharacterExtractor",
    "fourteen",
    "sixteen",
    "dislex",
]
