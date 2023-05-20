"""Semantics."""
from wordkit.features.semantics.embedding import EmbeddingTransformer
from wordkit.features.semantics.wordnet import HypernymSemanticsTransformer, OneHotSemanticsTransformer

__all__ = [
    "EmbeddingTransformer",
    "OneHotSemanticsTransformer",
    "HypernymSemanticsTransformer",
]
