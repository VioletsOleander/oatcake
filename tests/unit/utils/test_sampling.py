import pytest
import torch

from oatcake.utils.sampling import SamplingStrategy, sample_tokens


@pytest.mark.parametrize(
    ("token_logits", "expected"),
    [
        (torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([[1], [0]])),
        (torch.tensor([[0.6, 0.4], [0.3, 0.7]]), torch.tensor([[0], [1]])),
    ],
)
def test_sample_tokens_greedy(token_logits: torch.Tensor, expected: torch.Tensor) -> None:
    sampled = sample_tokens(token_logits, SamplingStrategy.GREEDY)
    assert torch.equal(sampled, expected), f"Expected {expected}, but got {sampled}"
