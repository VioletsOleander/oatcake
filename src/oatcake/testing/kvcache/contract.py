from typing import TYPE_CHECKING

import pytest
import torch

from .constants import KVSTATES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oatcake.interface import KVCache, KVState

__all__ = ["KVCacheContractTests"]


class KVCacheContractTests:
    """Contract tests for `KVCache` implementations.

    Provide test utility methods corresponding to the methods of `KVCache`,
    and a fixture for `KVState` sequences.

    `KVCache` implementations should utilize utility methods defined
    here to test whether they adhere to the expected behavior contracts.
    """

    @pytest.fixture(params=KVSTATES)
    def kv_states(self, request: pytest.FixtureRequest) -> list[KVState]:
        kv_states: list[KVState] = request.param
        return kv_states

    def update_test(self, kv_cache: KVCache, kv_states: Sequence[KVState]) -> None:
        num_tokens_new = kv_states[0].keys.size(-2)

        num_tokens_before = kv_cache.get_num_tokens()
        kv_states_before = kv_cache.get_kv_states()

        kv_cache.update(kv_states=kv_states)

        num_tokens_after = kv_cache.get_num_tokens()
        kv_states_after = kv_cache.get_kv_states()

        # Token numbers match
        assert num_tokens_after == num_tokens_before + num_tokens_new

        # New states match
        for (k_new, v_new), (k_after, v_after) in zip(
            kv_states, kv_states_after[-len(kv_states) :], strict=True
        ):
            assert torch.equal(k_after[:, :, -num_tokens_new:, :], k_new)
            assert torch.equal(v_after[:, :, -num_tokens_new:, :], v_new)

        # Old states are unchanged
        if num_tokens_before == 0:
            assert len(kv_states_before) == 0
        else:
            for (k_before, v_before), (k_after, v_after) in zip(
                kv_states_before, kv_states_after, strict=True
            ):
                assert k_after.size(-2) == k_before.size(-2) + num_tokens_new
                assert v_after.size(-2) == v_before.size(-2) + num_tokens_new
                assert torch.equal(k_after[:, :, :-num_tokens_new, :], k_before)
                assert torch.equal(v_after[:, :, :-num_tokens_new, :], v_before)

    def crop_test(self, kv_cache: KVCache, num_tokens_crop: int) -> None:
        num_tokens_before = kv_cache.get_num_tokens()
        kv_states_before = kv_cache.get_kv_states()

        kv_cache.crop(num_tokens_crop=num_tokens_crop)

        num_tokens_after = kv_cache.get_num_tokens()
        kv_states_after = kv_cache.get_kv_states()

        # Token numbers match
        assert num_tokens_after == num_tokens_before - num_tokens_crop

        # Remaining states match
        for (k_before, v_before), (k_after, v_after) in zip(
            kv_states_before, kv_states_after, strict=True
        ):
            assert k_after.size(-2) == k_before.size(-2) - num_tokens_crop
            assert v_after.size(-2) == v_before.size(-2) - num_tokens_crop
            assert torch.equal(k_after, k_before[:, :, :num_tokens_after, :])
            assert torch.equal(v_after, v_before[:, :, :num_tokens_after, :])
