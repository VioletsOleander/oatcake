from typing import TYPE_CHECKING

import pytest

from oatcake.inferencer.chunkwise import ChunkwiseDecodeInferencer
from oatcake.testing.inferencer.constants import MAX_NEW_TOKENS
from oatcake.testing.inferencer.contract import InferencerContractTests
from oatcake.testing.inferencer.scenario import InferenceScenarioFixtures

if TYPE_CHECKING:
    from oatcake.testing.inferencer.scenario import InferenceScenario
    from oatcake.utils.sampling import SamplingStrategy


@pytest.fixture(params=[1, 4, 8])
def decode_chunk_size(request: pytest.FixtureRequest) -> int:
    return request.param


class TestChunkwiseDecodeInferencer(InferencerContractTests, InferenceScenarioFixtures):
    def test_prefill(
        self, scenario: InferenceScenario, sampling_strategy: SamplingStrategy
    ) -> None:
        chunkwise_inferencer = ChunkwiseDecodeInferencer(model=scenario.fake_model)
        super().prefill_test(
            inferencer=chunkwise_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            sampling_strategy=sampling_strategy,
        )

    @pytest.mark.parametrize("max_new_tokens", MAX_NEW_TOKENS)
    def test_decode(
        self,
        scenario: InferenceScenario,
        max_new_tokens: int,
        sampling_strategy: SamplingStrategy,
        decode_chunk_size: int,
    ) -> None:
        chunkwise_inferencer = ChunkwiseDecodeInferencer(
            model=scenario.fake_model, decode_chunk_size=decode_chunk_size
        )
        super().decode_test(
            inferencer=chunkwise_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            max_new_tokens=max_new_tokens,
            sampling_strategy=sampling_strategy,
        )
