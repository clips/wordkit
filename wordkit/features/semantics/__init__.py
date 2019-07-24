"""Semantics."""
from .wordnet import OneHotSemanticsTransformer, HypernymSemanticsTransformer
from .embedding import EmbeddingTransformer

__all__ = ["EmbeddingTransformer",
           "OneHotSemanticsTransformer",
           "HypernymSemanticsTransformer"]
