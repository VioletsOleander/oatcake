"""Provide Qwen3 model implementation of `Model`."""

from functools import cached_property
from typing import TYPE_CHECKING, cast, override

from transformers import Qwen3ForCausalLM

from oatcake.interface import Model
from oatcake.kvcache.dynamic import DynamicCache

if TYPE_CHECKING:
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from oatcake.interface import KVCache

__all__ = ["ModelImpl", "Qwen3Model"]


class Qwen3Model(Model):
    """Qwen3Model wraps huggingface Qwen3 models, implementing `Model`.

    Attributes:
        hf_model (Qwen3ForCausalLM): The underlying huggingface Qwen3 model.
    """

    hf_model: Qwen3ForCausalLM

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.hf_model = Qwen3ForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype="auto"
        )

    @cached_property
    @override
    def eos_token_id(self) -> int:
        eos_token_id = self.hf_model.config.eos_token_id
        if eos_token_id is None:
            raise ValueError("The model config does not have an eos_token_id.")

        return eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Forward the underlying huggingface model.

        Expects `kv_cache` to be an instance of `DynamicCache`.

        Refers to `Model.forward` for more details.

        Raises:
            ValueError: If the model forward output does not contain logits.
        """
        if not isinstance(kv_cache, DynamicCache):
            raise TypeError("Qwen3Model only supports DynamicCache as KVCache.")

        input_ids = query_token_ids
        forward_out: CausalLMOutputWithPast = self.hf_model(
            input_ids=cast("torch.LongTensor", input_ids),
            logits_to_keep=0,  # keeps all logits
            use_cache=True,
            past_key_values=kv_cache.cache,
        )

        # Notice that the huggingface Qwen model will update the `kvcache.cache` instance in-place,
        # (which is a side effect of its Qwen3Attention module) so we don't need to
        # manually call `update` on `kv_cache` here.

        if forward_out.logits is None:
            raise ValueError("Model forward output does not contain logits.")

        return forward_out.logits


ModelImpl = Qwen3Model
