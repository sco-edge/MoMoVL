"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import replace
from functools import partial
from os.path import join
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union, Any,
)
from copy import deepcopy
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import einops
from transformers import PreTrainedModel, GenerationConfig, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config_molmoe import (
    ActivationType,
    BlockType,
    LayerNormType,
    VisionBackboneType,
    ImagePooling2DType,
    ImageProjectType, 
    AttentionType,
)


from .config_molmoe import (
    MolmoConfig,
    VisionBackboneConfig, ModelConfig
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


log = logging.getLogger(__name__)

# Import unchanged modules from original .modeling_molmoe
from .modeling_molmoe import (
    OLMoConfigurationError,
    ensure_finite_,
    BufferCache,
    _non_meta_init_device,
    Embedding,
    Dropout,
    RMSLayerNorm,
    get_causal_attention_bias,
    MolmoAttention,
    MolmoMLP,
    MolmoeMLP,
    MolmoOutput,
    MolmoVisionBackbone,
    MolmoPretrainedModel
)

from utils.bitwidth import get_bitwidth_assignment
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
import warnings, copy

class MolmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MolmoeMLP(config) for _ in range(self.num_experts)])
        self.ff_norm = RMSLayerNorm(
            config,
            size=config.d_model,
            eps=config.layer_norm_eps
        )

        self.quant_method = None
        self._is_quantized = False
        warnings.warn(
            "This model has been initialized with unquantized weights on a meta/placeholder device. "
            "You MUST call `model.language.model.layers[i].mlp.quantize_experts()` before inference.",
            UserWarning
        )

    def quantize_experts(self):
        expert_config = {
            '4': BaseQuantizeConfig(nbits=4, group_size=64),
            '2': BaseQuantizeConfig(nbits=2, group_size=32)
        }
        # Quantize routed_experts (INT2/INT4)
        q_experts = nn.ModuleList([nn.ModuleDict({'2': e, '4': copy.deepcopy(e)}) for e in self.experts])
        for idx in range(len(q_experts)):
            for bitwidth in ['4', '2']:
                expert = q_experts[idx][bitwidth]
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    quant_proj = HQQLinear(
                        linear_layer=getattr(expert, proj), 
                        quant_config=expert_config[bitwidth],
                        compute_dtype=torch.bfloat16,
                        device='cuda',
                        initialize=True,
                        del_orig=True)
                    setattr(expert, proj, quant_proj)
        self.experts = q_experts
        self._is_quantized = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.quant_method is not None, 'quant_method must be specified (APEX/Fate)'
        assert self._is_quantized, 'experts must be quantized before inference'

        hidden_states = self.ff_norm(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Get bit-width assignment for each expert
        bitwidth_assignment = get_bitwidth_assignment(selected_experts, routing_weights, 
                                                      num_experts=len(self.experts), quant_method=self.quant_method)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # skip less important experts [MoE-APEX] 
            if bitwidth_assignment[expert_idx] == '0':
                expert_layer = torch.zeros_like
            else:
                expert_layer = self.experts[expert_idx][bitwidth_assignment[expert_idx]]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MolmoDecoderLayer(nn.Module):
    """
    A base class for transformer block implementations.
    """
    def __init__(
        self, 
        layer_id: int, 
        config: MolmoConfig, 
        cache: BufferCache
    ):
        super().__init__()
        self.attn = MolmoAttention(config, cache)
        if getattr(config, "moe_num_experts", 0) > 0:
            self.mlp = MolmoeSparseMoeBlock(config)
        else:
            self.mlp = MolmoMLP(config)
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        if config.head_dim is None:
            assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(
            config.residual_dropout, 
            mask_p=config.response_residual_dropout
        ) 

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get query, key, value projections.
        shape:
            for regular attn q, k, v: (batch_size, seq_len, d_model)
            for multi-query attn q: (batch_size, seq_len, d_model)
                              k, v: (batch_size, seq_len, d_model // n_heads)
            for group query attn q: (batch_size, seq_len, d_model)
                            k, v: (batch_size, seq_len, d_model // n_kv_heads)
        """

        att, cache = self.attn(
            x, 
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=drop_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )
        x = x + self.dropout(att, drop_mask=drop_mask)
        og_x = x
        x, _ = self.mlp(x)
        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache


class MolmoModel(MolmoPretrainedModel):
    def __init__(
        self, 
        config: MolmoConfig, 
        init_params: bool = True
    ):
        super().__init__(config)
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        wte = None
        if self.config.additional_vocab_size is not None:
            wte = Embedding(
                config.embedding_size or config.vocab_size,
                config.additional_vocab_size,
                config.d_model,
                device=config.init_device,
                initializer_range=config.initializer_range,
                new_embed_initializer_range=config.new_embedding_init_range
            )
        else:
            wte=nn.Embedding(
                config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=RMSLayerNorm(
                    config, 
                    size=config.d_model,
                    eps=config.layer_norm_eps),
            )
        )

        layers = [
            MolmoDecoderLayer(i, config, self.__cache) \
                for i in range(config.n_layers)
        ]
        self.transformer.update({"blocks": nn.ModuleList(layers)})
 
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = MolmoVisionBackbone.build(config)

        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()


    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
    ) -> MolmoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.unconditioned and input_embeddings is None:
            images = None
            image_input_idx = None

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
        
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        num_image: Optional[int] = None
        if images is not None:
            # shape: (batch_size, num_image, num_patch, d_model)
            # cls_embed: (batch_size, num_image, d_model)
            image_features, cls_embed = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

            # inster the image feature into the embedding.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            # For hf demo/endpoint
            image_features = image_features.to(x.device)

            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

            if self.config.use_cls_feature:
                x = torch.cat([x[:, :1], cls_embed, x[:, 1:-num_image]], dim=1)
                
                valid_images = torch.any(
                    (image_input_idx >= 0).view(batch_size, num_image, num_patch), dim=-1
                )
                valid_images = valid_images.to(attention_mask.dtype)
                attention_mask = torch.cat(
                    [attention_mask[:, :1], valid_images, attention_mask[:, 1:-num_image]],
                    dim=1,
                )
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask, dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model ** 0.5)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        for block_idx, layer in enumerate(self.transformer.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]
            x, cache = layer(x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache)

            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        
        if images is not None and self.config.use_cls_feature:
            assert num_image is not None
            x = torch.cat(
                [x[:, :1], x[:, num_image+1:], torch.zeros_like(x[:, :num_image])],
                dim=1,
            )

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)
        
        # if self.config.scale_logits:
        #     logits.mul_(1 / math.sqrt(self.config.d_model))
        
        # if self.config.final_logit_softcapping is not None:
        #     logits = logits / self.config.final_logit_softcapping
        #     logits = torch.tanh(logits)
        #     logits = logits * self.config.final_logit_softcapping
        
        # if not last_logits_only and append_last_valid_logits is not None:
        #     last_valid_logit = logits[
        #         torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
        #     logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        return MolmoOutput(
            last_hidden_states=x,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) \
                if output_hidden_states else None
            )

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu",
        checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new


class MolmoForCausalLM_RT(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """
    config_class = MolmoConfig
    base_model_prefix = "model"
    _no_split_modules = ["MolmoDecoderLayer"]

    def __init__(
        self, 
        config: MolmoConfig
    ):
        super().__init__(config)
        # model_config = create_model_config_from_pretrained_config(config)
        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        config.init_device = "cpu"
        v_cfg = config.vision_backbone
        if v_cfg is not None:
            v_cfg = VisionBackboneConfig(**v_cfg)
            config.vision_backbone = v_cfg
        self.model = MolmoModel(config)

        if not config.weight_tying:
            self.lm_head = nn.Linear(
                config.d_model,
                config.embedding_size or config.vocab_size,
                bias=config.include_bias,
                device=config.init_device,
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        last_logits_only: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
        )
        
        x = outputs.last_hidden_states
        if self.config.weight_tying:
            logits = F.linear(x, self.model.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.lm_head(x)  # type: ignore

        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        
        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        loss = None
        if labels is not None:
            if loss_masks is not None:
                loss_masks = loss_masks * (loss_masks > 0)
                batch_size_in_tokens = max(loss_masks.sum().item(), 1)
                labels = labels.long()
                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1))
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits_for_loss, labels)
                loss = loss.view(input_ids.shape[0], -1)
                loss = loss * loss_masks
                loss = loss.sum() / batch_size_in_tokens
                use_zloss = getattr(self.config, "softmax_auxiliary_loss", False)
                if use_zloss:
                    z_squared = logits_for_loss.logsumexp(-1).pow(2)
                    z_loss = self.config.softmax_auxiliary_loss_scale * z_squared
                    z_loss = z_loss.view(input_ids.shape[0], -1)
                    z_loss = z_loss * loss_masks
                    z_loss = z_loss.sum() / batch_size_in_tokens
                    loss += z_loss
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.embedding_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=outputs.hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        images=None,
        attention_mask=None,
        image_masks=None,
        image_input_idx=None,
        generation_config=None,
        **kwargs,
    ):
        if generation_config is not None:
            assert generation_config.use_cache
 
        # images = batch.get("images")
        # image_masks = batch.get("image_masks")
        # image_input_idx = batch.get("image_input_idx")

        # Validate inputs.
        # input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        # attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        
        out = super().generate(
            # batch["input_ids"],
            input_ids,
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            position_ids=position_ids,
            append_last_valid_logits=append_last_valid_logits,
            **kwargs,
        )

        return out

    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        if generation_config is not None:
            assert generation_config.use_cache
        
        images = batch.get("images")
        image_masks = batch.get("image_masks")
        image_input_idx = batch.get("image_input_idx")

        # Validate inputs.
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        
        out = super().generate(
            batch["input_ids"],
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            position_ids=position_ids,
            append_last_valid_logits=append_last_valid_logits,
            **kwargs,
        )

        return out
        
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]

        if self.config.use_position_ids:
            attention_mask = kwargs.get("attention_mask")
            images = kwargs.get("images")
            image_masks = kwargs.get("image_masks")
            image_input_idx = kwargs.get("image_input_idx")
            position_ids = kwargs.get("position_ids")
            append_last_valid_logits = kwargs.get("append_last_valid_logits")
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "last_logits_only": True,
            }
            if past_key_values is None:
                model_inputs["images"] = images
                model_inputs["image_masks"] = image_masks
                model_inputs["image_input_idx"] = image_input_idx
                model_inputs["append_last_valid_logits"] = append_last_valid_logits
        else:    
            model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

            model_inputs.update(kwargs)
            model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if self.config.use_position_ids:
            model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
            if "append_last_valid_logits" in model_kwargs:
                del model_kwargs["append_last_valid_logits"]
            if "images" in model_kwargs:
                del model_kwargs["images"]
                del model_kwargs["image_masks"]
                del model_kwargs["image_input_idx"]
        try:
            # [DEV] Patch for transformers 4.38.2
            # cache_name, cache = super()._extract_past_from_model_output(outputs)
            cache = super()._extract_past_from_model_output(outputs)
            cache_name = "past_key_values"
        except AttributeError:
            past_key_values = outputs.past_key_values if "past_key_values" in outputs else None
            cache_name, cache = "past_key_values", past_key_values

        model_kwargs[cache_name] = cache
        # [DEV] Patch for transformers 4.38.2
        if "cache_position" in model_kwargs:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            # Fallback for older transformers that don't track cache_position
            model_kwargs["cache_position"] = torch.tensor([model_kwargs.get("past_key_values_length", 0)])

        return model_kwargs

    # TODO: these are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.lm_head

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.lm_head = value

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `lm_head` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.embedding_size = model_embeds.weight.shape[0]
        self.model.config.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.embedding_size < self.config.vocab_size:
            warning_message = (
                f"Resizing token embeddings to size {self.config.embedding_size}, which is less than the vocab size "
                f"{self.config.vocab_size} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds
