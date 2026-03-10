from typing import NamedTuple

from oatcake.testing.model.fake import FakeModelConfig

__all__ = ["FAKE_MODEL_CONFIGS", "MAX_NEW_TOKENS", "QUERY_SHAPES", "QueryShape"]

FAKE_MODEL_CONFIGS = (
    FakeModelConfig(eos_token_id=0, vocab_size=2, num_layers=1, num_heads=2, embed_dim=4),
    FakeModelConfig(eos_token_id=1, vocab_size=5, num_layers=2, num_heads=4, embed_dim=8),
    FakeModelConfig(eos_token_id=2, vocab_size=8, num_layers=4, num_heads=6, embed_dim=12),
)


class QueryShape(NamedTuple):
    batch_size: int
    num_query_tokens: int


QUERY_SHAPES = (
    QueryShape(batch_size=1, num_query_tokens=9),
    QueryShape(batch_size=1, num_query_tokens=6),
    QueryShape(batch_size=1, num_query_tokens=3),
)

MAX_NEW_TOKENS = (1, 3, 6)
