"""Provide `SampleStrategy`, `sample_tokens` for token generation."""

from enum import StrEnum, auto

import torch

__all__ = ["SamplingStrategy", "sample_tokens"]


class SamplingStrategy(StrEnum):
    """Sampling strategies for token generation.

    Attributes:
        RANDOM: Sample tokens probabilistically according to the token distribution over vocabulary.
        GREEDY: Always select the token with the highest probability (argmax).
    """

    RANDOM = auto()
    GREEDY = auto()


def sample_tokens(token_logits: torch.Tensor, sampling_strategy: SamplingStrategy) -> torch.Tensor:
    """Sample token ids with `token_logits` according to `sampling_strategy`.

    `token_logits` is expected to be raw logits.

    Args:
        token_logits (torch.Tensor): Logits of shape `[batch_size, vocab_size]`.
        sampling_strategy (SamplingStrategy): Sampling strategy to use.

    Returns:
        torch.Tensor: Sampled next token ids of shape `[batch_size, 1]`.
    """
    match sampling_strategy:
        case SamplingStrategy.GREEDY:
            token_ids = torch.argmax(token_logits, dim=-1, keepdim=True)
        case SamplingStrategy.RANDOM:
            probs = torch.softmax(token_logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1)
        case _:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    return token_ids
