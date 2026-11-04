from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from glob import glob
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from transformers import PretrainedConfig


C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


PathOrStr = Union[str, PathLike]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"



class AttentionType(StrEnum):
    sdpa = "sdpa"
    direct = "direct"
    flash = "flash"


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """

    gemma_rms = "gemma_rms"
    """
    A GemmaRMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    quick_gelu = "quick_gelu"
    gelu = "gelu"
    gelu_tanh = "gelu_tanh"
    relu = "relu"
    silu = "silu"
    llama_geglu = "llama_geglu"
    llama_geglu_tanh = "llama_geglu_tanh"
    llama_swiglu = "llama_swiglu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """

    gemma = "gemma"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Gemma.
    """

    moe = "moe"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class VisionBackboneType(StrEnum):
    openai = "openai"


class ImagePaddingEmbed(StrEnum):
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    attention = "attention"
    attention_meanq = "attention-meanq"
    attention_2wide = "attention_2wide"
    attention_v2 = "attention-v2"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


@dataclass
class VisionBackboneConfig:
    image_model_type: VisionBackboneType = VisionBackboneType.openai
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 24
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: ActivationType = ActivationType.gelu
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    fsdp_wrap: bool = False

    # how to preprocess imagse for this ViT
    resize_mode: str = "default"

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class ModelConfig:
    """
    OLMo (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for Llama2-style grouped query attention.
    """

    qkv_bias: bool = False  # qwen models use bias in kvq layers

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.
    """

    alibi: bool = False
    """
    If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: float = 10000.

    rope_impl: str = "cockatoo"

    vit_load_path: Optional[str] = None
    """
    Use this to load the vit model.
    """

    llm_load_path: Optional[str] = None
    """
    Use this to partially load the llm transformer.
    """

    low_cpu_fsdp: bool = True
    """
    If ``True``, we save cpu memory by loading the pretrained vision model on randk0 only
    when init_device is `meta`.
    If TrainConfig.load_path is set, this should be set to ``False`` (default: True)
    """

    attention_type: AttentionType = AttentionType.sdpa
    """
    Attention implementation to use.
    """

    float32_attention: bool = True
    """
    Compute attention in float32
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    # Only apply dropout to response tokens
    response_attention_dropout: float = 0.0

    multi_query_attention: Optional[bool] = None
    """
    Deprecated. Use n_kv_heads instead.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    # Only apply dropout to response tokens
    response_residual_dropout: float = 0.0

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    layer_norm_eps: Optional[float] = None

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    max_position_embeddings: Optional[int] = None

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large
    models tend to have near 0 bias terms anyway.
    """

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
    layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    scale_logits: bool = False
    """
    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    # For new special tokens
    additional_vocab_size: Optional[int] = None

    new_embedding_init_range: float = 0.02
    """
    How to initialize embedding for new 
    """

    weight_tying: bool = True
    """
    Whether to tie output linear weights to the input embedding.
    """

    pad_token_id: int = -1
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_fn: InitFnType = InitFnType.normal
    """
    The weight initialization strategy.
    """

    init_std: float = 0.02
    """
    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    norm_after: bool = False
    """
    Apply norm after the attention/feedforward layers rather than before, as introduced in the Swin transformer paper (Liu et al).
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """

    moe_num_experts: Optional[int] = 8
    """
    The number of experts to use in the MoE block.
    """

    moe_top_k: Optional[int] = 2
    """
    The number of experts to select for each token.
    """

    moe_mlp_impl: Optional[str] = "sparse"
    """
    Choose "grouped" for grouped GEMM installable via `pip install git+https://git@github.com/tgale96/grouped_gemm.git@66c7195e35e8c4f22fa6a014037ef511bfa397cb`.
    """

    moe_log_expert_assignment: Optional[bool] = False
    """
    Whether to log the expert assignment.
    """

    moe_shared_expert: Optional[bool] = False
    """
    Whether to have an always-used expert like in [DeepSeekMoE](https://arxiv.org/abs/2401.06066).
    """

    moe_lbl_in_fp32: Optional[bool] = False
    """
    Whether to perform load balancing in FP32.
    """

    moe_interleave: Optional[bool] = False
    """
    Interleave sequential with MoE blocks starting with sequential.
    """

    moe_loss_weight: Optional[float] = 0.1
    """
    The weight to use for the MoE load balancing loss.
    """

    moe_zloss_weight: Optional[float] = None
    """
    Weight for MoE router z-loss where None means no router z-loss. 0.001 is a common value.
    """

    moe_dropless: Optional[bool] = True
    """
    Whether to use [dMoE](https://arxiv.org/abs/2211.15841).
    """

    moe_capacity_factor: Optional[float] = 1.25
    """
    The capacity factor to use in the MoE block. Only applies if not using dMoE.
    """

    # Image pre-processing options.
    max_crops: int = 12

    crop_mode: str = "patchify-v2-and-resize-c2"

    do_random_scale: bool = True

    use_col_tokens: bool = True

    # How to prompt the model
    prompt_type: str = "none"

    # System prompt to use
    system_prompt_kind: str = "style"

    # How to format messages
    message_formatting: str = "none"

    always_start_with_space: bool = True

    prompt_override: Optional[str] = None

    default_inference_len: Optional[int] = 65

    overlap_margins: Tuple[int, int] = (4, 4)

    image_padding_embed: Optional[ImagePaddingEmbed] = None

    # What layers to get from the image encoder
    vit_layers: Tuple = (-1,)

    # Controls the image/language connector
    image_pooling_h: int = 2

    image_pooling_w: int = 2

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention

    image_projector: ImageProjectType = ImageProjectType.mlp

    image_feature_dropout: float = 0.0

    use_cls_feature: bool = False

    fix_image_input_idx: int = 2

    # Makes the model ignore the image
    unconditioned: bool = False

    # Use in combination with sub-sequence experts to make imags/text tokens always
    # occupy particular sub-sequences of the input
    pad_to: Optional[int] = None

    # LLM Transformer settings
    initializer_range: float = 0.02

    pad_tokenizer: bool = False

    normalize_input_embeds: bool = False

    use_position_ids: bool = True
    """
    Whether to use position IDs in the model.
    The model operation regarding positional embeddings changes depending on this variable.
    """

    query_pre_attn_scalar: int = 224
    """
    Scalar to apply to the queries before attention.
    Used for Gemma-2.
    """

    attn_logit_softcapping: Optional[float] = None
    """
    Softcap the logits in the attention mechanism.
    Used for Gemma-2.
    """

    final_logit_softcapping: Optional[float] = None
    """
    Softcap the final logits.
    Used for Gemma-2.
    """

    head_dim: Optional[int] = None
    """
    The head dimensionality for the attention mechanism.
    Used for Gemma-2.
    """

    loss_token_weighting: Optional[str] = None

    gin_bindings: Optional[str] = None


class MolmoConfig(PretrainedConfig):
    model_type = "molmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = asdict(model_config)
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["OLMoForCausalLM"]) or ["OLMoForCausalLM"]}
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def image_num_patch(self):
        h, w = (336, 336)
        return h // 14, w // 14
     
    @property
    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise ValueError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )

