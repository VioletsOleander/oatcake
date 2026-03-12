"""Define `KVCache` and `KVState`."""

from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

__all__ = ["KVCache", "KVState"]


class KVState(NamedTuple):
    """Keys and values tensor for a single transformer layer.

    The shape of `keys` and `values` are
    `[batch_size, num_attention_heads, num_tokens, head_dim]`.

    Attributes:
        keys (torch.Tensor): The keys tensor.
        values (torch.Tensor): The values tensor.
    """

    keys: torch.Tensor
    values: torch.Tensor


class KVCache(Protocol):
    """Stores layerwise key and value tensors, which are used by an Inferencer during inference."""

    def update(self, kv_states: Iterable[KVState]) -> None:
        """Update the storage with new key and value tensors.

        The length of `kv_states` should be equal to the number of transformer layers,
        and each `KVState` contains the new key and value tensors for the corresponding layer.

        All key and value tensors in `kv_states` should have the same shape.

        Args:
            kv_states (Iterable[KVState]): Iterable of KVState, where each KVState contains
                the new key and value tensors for the corresponding layer.
        """
        ...

    def update_layer(self, kv_state: KVState, layer_idx: int) -> None:
        """Update the storage with new key and value tensors for a specific layer.

        `kv_state` contains the new key and value tensors for the layer specified by `layer_idx`.
        `layer_idx` should be a valid index, i.e. range from 0 to the number of transformer layers - 1.

        The shape of key and value tensors in `kv_state` should be the same as those already stored
        for the same layer (if any).

        Args:
            kv_state (KVState): A KVState containing the new key and value tensors for the layer.
            layer_idx (int): The index of the layer to update, should be non-negative.
        """
        ...

    def crop(self, num_tokens_crop: int) -> None:
        """Crop the latest `num_tokens_crop` tokens from the cache.

        If `num_tokens_crop` is non-positive, the cache will remain unchanged.
        If `num_tokens_crop` exceeds the number of currently cached tokens, all
        cached tokens will be cropped.

        Args:
            num_tokens_crop (int): Number of latest tokens to crop from the cache.
        """
        ...

    def get_num_tokens(self) -> int:
        """Get the current number of tokens stored in the cache.

        Returns:
            int: Current number of tokens in the cache.
        """
        ...

    def get_kv_states(self) -> list[KVState] | None:
        """Get the current stored KV states for all layers.

        Returns:
            list[KVState] | None: A list of KVState. If the cache is empty, None is returned.
        """
        ...

    def get_kv_state(self, layer_idx: int) -> KVState | None:
        """Get the current stored KV state for a specific layer.

        Args:
            layer_idx (int): The index of the layer to get the KV state for, should be non-negative.

        Returns:
            KVState | None: The KVState for the specified layer.
                If the cache is empty or the layer index is out of range, None is returned.
        """
        ...
