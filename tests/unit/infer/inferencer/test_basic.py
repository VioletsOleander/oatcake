from typing import TYPE_CHECKING

import pytest

from oatcake.inferencer.basic import BasicInferencer
from oatcake.testing.inferencer.constants import MAX_NEW_TOKENS
from oatcake.testing.inferencer.contract import InferencerContractTests
from oatcake.testing.inferencer.scenario import InferenceScenarioFixtures

if TYPE_CHECKING:
    from oatcake.testing.inferencer.scenario import InferenceScenario
    from oatcake.utils.sampling import SamplingStrategy


class TestBasicInferencer(InferencerContractTests, InferenceScenarioFixtures):
    def test_prefill(
        self, scenario: InferenceScenario, sampling_strategy: SamplingStrategy
    ) -> None:
        basic_inferencer = BasicInferencer(model=scenario.fake_model)
        super().prefill_test(
            inferencer=basic_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            sampling_strategy=sampling_strategy,
        )

    @pytest.mark.parametrize("max_new_tokens", MAX_NEW_TOKENS)
    def test_decode(
        self, scenario: InferenceScenario, max_new_tokens: int, sampling_strategy: SamplingStrategy
    ) -> None:
        basic_inferencer = BasicInferencer(model=scenario.fake_model)
        super().decode_test(
            inferencer=basic_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            max_new_tokens=max_new_tokens,
            sampling_strategy=sampling_strategy,
        )
