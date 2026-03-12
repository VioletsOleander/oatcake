"""Define `DynamicCache`, implementing `KVCache`."""

from collections.abc import Iterable, Sequence
from typing import overload, override

import torch

from oatcake.interface import KVCache, KVState

__all__ = ["DynamicCache", "KVCacheImpl"]


class DynamicCache(KVCache, Sequence):
    """DynamicCache implements `KVCache` and `Sequence`.

    Attributes:
        kv_states (list[KVState]): List of KVState containing the past key and value tensors for
            each transformer layer.
    """

    kv_states: list[KVState]

    def __init__(self) -> None:
        self.kv_states = []

    @override
    def update(self, kv_states: Iterable[KVState]) -> None:
        for layer_idx, kv_state in enumerate(kv_states):
            self.update_layer(kv_state, layer_idx)

    @override
    def update_layer(self, kv_state: KVState, layer_idx: int) -> None:
        # Initialize on demand if the layer index is out of current range
        while len(self.kv_states) <= layer_idx:
            self.kv_states.append(KVState(keys=torch.empty(0), values=torch.empty(0)))

        state = self.kv_states[layer_idx]
        self.kv_states[layer_idx] = KVState(
            keys=torch.cat((state.keys, kv_state.keys), dim=2),
            values=torch.cat((state.values, kv_state.values), dim=2),
        )

    @override
    def crop(self, num_tokens_crop: int) -> None:
        if num_tokens_crop <= 0:
            return

        if num_tokens_crop >= self.get_num_tokens():
            self.kv_states = []
            return

        for layer_idx, state in enumerate(self.kv_states):
            self.kv_states[layer_idx] = KVState(
                keys=state.keys[..., :-num_tokens_crop, :],
                values=state.values[..., :-num_tokens_crop, :],
            )

    @override
    def get_num_tokens(self) -> int:
        return self.kv_states[0].keys.shape[2] if self.kv_states else 0

    @override
    def get_kv_states(self) -> list[KVState] | None:
        return self.kv_states.copy() if self.kv_states else None

    @override
    def get_kv_state(self, layer_idx: int) -> KVState | None:
        if 0 <= layer_idx < len(self.kv_states):
            return self.kv_states[layer_idx]
        return None

    @overload
    def __getitem__(self, index: int) -> KVState: ...

    @overload
    def __getitem__(self, index: slice) -> list[KVState]: ...

    @override
    def __getitem__(self, index: int | slice) -> KVState | list[KVState]:
        return self.kv_states[index]

    @override
    def __len__(self) -> int:
        return len(self.kv_states)


KVCacheImpl = DynamicCache
