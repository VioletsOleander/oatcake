"""Provide `Transformer`, which is a simple implementation of decoder-only transformer Model."""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Transformer"]


class _Attention(nn.Module):
    """A simple implementation of scaled dot-product attention."""

    embed_dim: int
    num_heads: int
    head_dim: int
    q_linear: nn.Linear
    k_linear: nn.Linear
    v_linear: nn.Linear
    out_linear: nn.Linear

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim, remainder = divmod(embed_dim, num_heads)
        if remainder != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, but got embed_dim={embed_dim} and num_heads={num_heads}"
            )

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, embedding: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute attention output for the given query and KV cache.

        Args:
            embedding (torch.Tensor): Input embedding of shape (batch_size, seq_len, embed_dim).
            kv_cache (tuple[torch.Tensor, torch.Tensor] | None): Tuple of cached key and value tensors,
                or None if no cache is available.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: Attention output of shape
                (batch_size, seq_len, embed_dim) and newly computed tuple of key and value tensors
                for the current input.
        """
        batch_size, num_q_tokens, _ = embedding.size()
        device = embedding.device

        # KQV projections
        q: torch.Tensor = self.q_linear(embedding)
        q = q.view(batch_size, num_q_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k: torch.Tensor = self.k_linear(embedding)
        k = k.view(batch_size, num_q_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v: torch.Tensor = self.v_linear(embedding)
        v = v.view(batch_size, num_q_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        query = q
        key = k if kv_cache is None else torch.cat([kv_cache[0], k], dim=2)
        value = v if kv_cache is None else torch.cat([kv_cache[1], v], dim=2)

        # Build the lower-right causal mask so that query position i attends to all
        # key positions up to (S - L + i), which is correct for both full-sequence
        # (L == S, no cache) and incremental decoding (L < S, with KV cache).
        # Using is_causal=True alone would apply the "upper-left" convention and
        # restrict query[i] to attend only to key[i], ignoring all cached keys.
        num_k_tokens = key.size(2)
        num_cached_tokens = num_k_tokens - num_q_tokens  # num_cached_tokens = L - S
        q_indices = torch.arange(num_q_tokens, device=device).unsqueeze(1)  # [num_q_tokens, 1]
        k_indices = torch.arange(num_k_tokens, device=device).unsqueeze(0)  # [1, num_k_tokens]
        attn_mask = k_indices <= q_indices + num_cached_tokens

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=key, value=value, attn_mask=attn_mask
        )

        # Out projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_q_tokens, self.embed_dim)
        return self.out_linear(attn_output), (k, v)


class _TransformerBlock(nn.Module):
    """A simple implementation of a transformer block, consisting of a self-attention layer and a feedforward layer."""

    attention: _Attention
    feedforward: nn.Sequential
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = _Attention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, embedding: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward the input embedding through the transformer block.

        Args:
            embedding (torch.Tensor): Input embedding of shape (batch_size, seq_len, embed_dim).
            kv_cache (tuple[torch.Tensor, torch.Tensor] | None): Tuple of cached key and value tensors,
                or None if no cache is available.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: Output embedding of shape (batch_size, seq_len, embed_dim)
                and newly computed key and value tensors for the current input.
        """
        attn_output, new_kv_state = self.attention(embedding, kv_cache)
        x = self.norm1(embedding + attn_output)
        ff_output = self.feedforward(x)
        return self.norm2(x + ff_output), new_kv_state


class Transformer(nn.Module):
    """A simple implementation of a transformer model, consisting of an embedding layer and multiple transformer blocks."""

    embedding: nn.Embedding
    blocks: nn.ModuleList
    lm_head: nn.Linear

    def __init__(self, vocab_size: int, num_blocks: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward the input token ids through the transformer model.

        The `kv_cache` will be updated with the newly computed KV states as a side effect.

        Args:
            query_token_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            kv_cache (Sequence[tuple[torch.Tensor, torch.Tensor]] | None): Sequence of tuples containing
                cached key and value tensors, or None if no cache is available.

        Returns:
            tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]: Output logits of shape
                (batch_size, seq_len, vocab_size) and the newly computed list of key and value tensors
                for the current input.
        """
        embedding: torch.Tensor = self.embedding(query_token_ids)

        kv_states = kv_cache if kv_cache is not None else [None] * len(self.blocks)
        new_kv_states: list[tuple[torch.Tensor, torch.Tensor]] = []

        for block, kv_state in zip(self.blocks, kv_states, strict=True):
            embedding, new_kv_state = block(embedding, kv_state)
            new_kv_states.append(new_kv_state)

        return self.lm_head(embedding), new_kv_states
