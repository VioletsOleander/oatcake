"""Provide `FakeModel` and its configuration `FakeModelConfig`."""

from typing import TYPE_CHECKING, NamedTuple, override

from oatcake.interface import KVCache, KVState, Model

from .transformer import Transformer

if TYPE_CHECKING:
    import torch

__all__ = ["FakeModel", "FakeModelConfig"]


class FakeModelConfig(NamedTuple):
    """Configuration parameters for `FakeModel`.

    Attributes:
        eos_token_id (int): End-of-sequence token id.
        vocab_size (int): Number of tokens in the vocabulary.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads for multi-head attention.
        embed_dim (int): Dimension of the token embeddings.
    """

    eos_token_id: int
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int


class FakeModel(Model):
    """Lightweight fake implementation of `Model`.

    Contains minimal logic to simulate the behavior of a real model.

    Attributes:
        config (FakeModelConfig): Configuration parameters for the fake model.
        transformer (Transformer): Underlying transformer model to generate token logits.
    """

    config: FakeModelConfig
    transformer: Transformer

    def __init__(self, config: FakeModelConfig) -> None:
        self.config = config

        self.transformer = Transformer(
            vocab_size=config.vocab_size,
            num_blocks=config.num_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
        )

    @property
    @override
    def eos_token_id(self) -> int:
        return self.config.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Perform a forward pass through the fake model."""
        kv_states = kv_cache.get_kv_states() or None

        logits, new_kv_states = self.transformer.forward(
            query_token_ids=query_token_ids, kv_cache=kv_states
        )

        new_kv_states = map(KVState._make, new_kv_states)
        kv_cache.update(kv_states=new_kv_states)

        return logits
