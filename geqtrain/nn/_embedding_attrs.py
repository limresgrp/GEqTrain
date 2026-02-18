"""Compatibility wrapper for embedding attribute modules.

New code lives in ``geqtrain.nn.embeddings``.
This module is kept to preserve existing imports and Hydra targets.
"""

from geqtrain.nn.embeddings import (
    EmbeddingInputAttrs,
    EmbeddingAttrs,
    BaseNodeEmbedding,
    BaseNodeEqEmbedding,
    OneHotEncoding,
    apply_masking,
)

__all__ = [
    "EmbeddingInputAttrs",
    "EmbeddingAttrs",
    "BaseNodeEmbedding",
    "BaseNodeEqEmbedding",
    "OneHotEncoding",
    "apply_masking",
]
