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

    def assert_equal(
        self, x: Sequence[KVState], y: Sequence[KVState], x_range: slice, y_range: slice
    ) -> None:
        for (k1, v1), (k2, v2) in zip(x, y, strict=True):
            assert torch.equal(k1[..., x_range, :], k2[..., y_range, :])
            assert torch.equal(v1[..., x_range, :], v2[..., y_range, :])

    def update_test(self, kv_cache: KVCache, kv_states: Sequence[KVState]) -> None:
        num_tokens_new = kv_states[0].keys.size(-2)

        num_tokens_before = kv_cache.get_num_tokens()
        kv_states_before = kv_cache.get_kv_states()

        kv_cache.update(kv_states=kv_states)

        num_tokens_after = kv_cache.get_num_tokens()
        kv_states_after = kv_cache.get_kv_states()

        # Token numbers match
        assert num_tokens_after == num_tokens_before + num_tokens_new

        if num_tokens_before == 0 and num_tokens_after == 0:
            assert kv_states_before is None
            assert kv_states_after is None
        elif num_tokens_before == 0 and num_tokens_after > 0:
            assert kv_states_before is None
            assert kv_states_after is not None
            # New states match
            self.assert_equal(
                x=kv_states_after,
                y=kv_states,
                x_range=slice(None, None),
                y_range=slice(None, None),
            )
        elif num_tokens_before > 0 and num_tokens_after > 0:
            assert kv_states_before is not None
            assert kv_states_after is not None
            # Previous states match
            self.assert_equal(
                x=kv_states_before,
                y=kv_states_after,
                x_range=slice(None, None),
                y_range=slice(None, num_tokens_before),
            )
            # New states match
            self.assert_equal(
                x=kv_states_after,
                y=kv_states,
                x_range=slice(num_tokens_before, None),
                y_range=slice(None, None),
            )
        else:
            pytest.fail("Unexpected token number scenario")

    def crop_test(self, kv_cache: KVCache, num_tokens_crop: int) -> None:
        num_tokens_before = kv_cache.get_num_tokens()
        kv_states_before = kv_cache.get_kv_states()

        kv_cache.crop(num_tokens_crop=num_tokens_crop)

        num_tokens_after = kv_cache.get_num_tokens()
        kv_states_after = kv_cache.get_kv_states()

        # Token numbers match
        assert num_tokens_after == num_tokens_before - num_tokens_crop

        if num_tokens_before == 0 and num_tokens_after == 0:
            assert kv_states_before is None
            assert kv_states_after is None
        elif num_tokens_before > 0 and num_tokens_after == 0:
            assert kv_states_before is not None
            assert kv_states_after is None
        elif num_tokens_before > 0 and num_tokens_after > 0:
            assert kv_states_before is not None
            assert kv_states_after is not None
            # Remaining states match
            self.assert_equal(
                x=kv_states_before,
                y=kv_states_after,
                x_range=slice(None, num_tokens_before - num_tokens_crop),
                y_range=slice(None, None),
            )
        else:
            pytest.fail("Unexpected token number scenario")
