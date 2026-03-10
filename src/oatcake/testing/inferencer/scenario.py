"""Define `InferenceScenario` and `InferenceScenarioFixtures` for environment setup of inference tests."""

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

from oatcake.kvcache.dynamic import DynamicCache
from oatcake.testing.model.fake import FakeModel

from .constants import FAKE_MODEL_CONFIGS, QUERY_SHAPES

if TYPE_CHECKING:
    from oatcake.testing.model.fake import FakeModelConfig

    from .constants import QueryShape

__all__ = ["InferenceScenario", "InferenceScenarioFixtures"]


class InferenceScenario(NamedTuple):
    query_token_ids: torch.Tensor
    fake_model: FakeModel
    kv_cache: DynamicCache


class InferenceScenarioFixtures:
    @pytest.fixture(params=QUERY_SHAPES)
    def query_shape(self, request: pytest.FixtureRequest) -> QueryShape:
        return request.param

    @pytest.fixture(params=FAKE_MODEL_CONFIGS)
    def model_config(self, request: pytest.FixtureRequest) -> FakeModelConfig:
        return request.param

    @pytest.fixture
    def scenario(self, query_shape: QueryShape, model_config: FakeModelConfig) -> InferenceScenario:
        query_token_ids = torch.randint(low=0, high=model_config.vocab_size, size=query_shape)
        fake_model = FakeModel(config=model_config)
        kv_cache = DynamicCache()

        return InferenceScenario(
            query_token_ids=query_token_ids, fake_model=fake_model, kv_cache=kv_cache
        )
