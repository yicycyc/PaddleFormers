# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle Pixtral Vision model."""

import paddle
from paddle import nn

from ..cache_utils import Cache
from ..llama.modeling import LlamaModel
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import PixtralConfig, PixtralVisionConfig


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    """Compute position ids for 2D meshgrid of patches."""
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        h_grid = paddle.arange(height).unsqueeze(1).expand([height, width]).reshape([-1])
        v_grid = paddle.arange(width).unsqueeze(0).expand([height, width]).reshape([-1])
        ids = h_grid * max_width + v_grid
        positions.append(ids)
    return paddle.concat(positions)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralRotaryEmbedding(nn.Layer):
    """
    2D Rotary Position Embedding for Pixtral vision encoder.

    For each pixel position in the 2D grid (height x width), computes a unique
    frequency embedding using different frequencies for height and width dimensions.
    """

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters.get("rope_type", "default")
        if self.rope_type != "default":
            raise ValueError(
                f"{self.__class__.__name__} does not support non-default RoPE, but got `rope_type={self.rope_type}`"
            )

        inv_freq = self._compute_inv_freq(config)
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    @staticmethod
    def _compute_inv_freq(config: PixtralVisionConfig):
        """Compute 2D inverse frequencies for the vision encoder."""
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim

        max_patches_per_side = config.image_size // config.patch_size
        h = paddle.arange(max_patches_per_side)
        w = paddle.arange(max_patches_per_side)

        freqs = 1.0 / (base ** (paddle.arange(0, dim, 2).astype("float32") / dim))
        freqs_h = paddle.outer(h.astype("float32"), freqs[::2])
        freqs_w = paddle.outer(w.astype("float32"), freqs[1::2])

        inv_freq = paddle.concat(
            [
                freqs_h[:, None, :].expand([-1, max_patches_per_side, -1]),
                freqs_w[None, :, :].expand([max_patches_per_side, -1, -1]),
            ],
            axis=-1,
        ).reshape([-1, dim // 2])

        # cat to get full dim (same as llama format: cat(freqs, freqs))
        inv_freq = paddle.concat([inv_freq, inv_freq], axis=-1)
        return inv_freq

    @paddle.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: Input tensor, used only to determine dtype.
            position_ids: [batch_size, seq_len] position indices into the 2D grid.
        Returns:
            (cos, sin) tuple, each of shape [batch_size, seq_len, dim].
        """
        freqs = self.inv_freq[position_ids]
        with paddle.amp.auto_cast(enable=False):
            emb = freqs.astype("float32")
            cos = emb.cos()
            sin = emb.sin()
        return cos.astype(x.dtype), sin.astype(x.dtype)


class PixtralRMSNorm(nn.Layer):
    """RMS Normalization for Pixtral."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class PixtralMLP(nn.Layer):
    """MLP for Pixtral (gate + up + down projections)."""

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

        if config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_act == "silu":
            self.act_fn = nn.Silu()
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PixtralAttention(nn.Layer):
    """Multi-headed attention for Pixtral vision encoder."""

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        batch_size, patches, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        key_states = key_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        # Scaled dot-product attention
        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(
            query_states.dtype
        )
        if self.training and self.dropout > 0:
            attn_weights = paddle.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, patches, -1])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class PixtralAttentionLayer(nn.Layer):
    """A single transformer layer for Pixtral."""

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PixtralTransformer(nn.Layer):
    """Transformer encoder stack for Pixtral."""

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList([PixtralAttentionLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
            )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


def generate_block_attention_mask(patch_embeds_list, tensor):
    """Generate block attention mask so patches from different images don't attend to each other."""
    dtype = tensor.dtype
    seq_len = tensor.shape[1]
    d_min = paddle.finfo(dtype).min
    causal_mask = paddle.full([seq_len, seq_len], fill_value=d_min, dtype=dtype)

    block_end_idx = paddle.to_tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = paddle.to_tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx.numpy().tolist(), block_end_idx.numpy().tolist()):
        causal_mask[int(start) : int(end), int(start) : int(end)] = 0

    causal_mask = causal_mask[None, None, :, :].expand([tensor.shape[0], 1, -1, -1])
    return causal_mask


class PixtralPretrainedModel(PretrainedModel):
    config_class = PixtralVisionConfig
    base_model_prefix = "vision_encoder"
    supports_gradient_checkpointing = True

    # Weight keys that need transposition when loading from PyTorch
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: PixtralVisionConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"vision_encoder.patch_conv.weight -> {model_prefix}patch_conv.weight",
            f"vision_encoder.ln_pre.weight -> {model_prefix}ln_pre.weight",
        ]

        # Transformer layers
        for layer_name in ["attention_norm", "ffn_norm"]:
            aoa_statements.append(
                f"vision_encoder.transformer.layers.$LAYER_ID.{layer_name}.weight -> "
                f"{model_prefix}transformer.layers.$LAYER_ID.{layer_name}.weight"
            )

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            aoa_statements.append(
                f"vision_encoder.transformer.layers.$LAYER_ID.attention.{proj_name}.weight^T -> "
                f"{model_prefix}transformer.layers.$LAYER_ID.attention.{proj_name}.weight"
            )

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            aoa_statements.append(
                f"vision_encoder.transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight^T -> "
                f"{model_prefix}transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight"
            )

        return aoa_statements

    @classmethod
    def _gen_inv_aoa_config(cls, config: PixtralVisionConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"{model_prefix}patch_conv.weight -> vision_encoder.patch_conv.weight",
            f"{model_prefix}ln_pre.weight -> vision_encoder.ln_pre.weight",
        ]

        for layer_name in ["attention_norm", "ffn_norm"]:
            aoa_statements.append(
                f"{model_prefix}transformer.layers.$LAYER_ID.{layer_name}.weight -> "
                f"vision_encoder.transformer.layers.$LAYER_ID.{layer_name}.weight"
            )

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            aoa_statements.append(
                f"{model_prefix}transformer.layers.$LAYER_ID.attention.{proj_name}.weight -> "
                f"vision_encoder.transformer.layers.$LAYER_ID.attention.{proj_name}.weight^T"
            )

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            aoa_statements.append(
                f"{model_prefix}transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight -> "
                f"vision_encoder.transformer.layers.$LAYER_ID.feed_forward.{proj_name}.weight^T"
            )

        return aoa_statements


class PixtralVisionModel(PixtralPretrainedModel):
    """Pixtral Vision Encoder with 2D Rotary Position Embedding."""

    def __init__(self, config: PixtralVisionConfig):
        super().__init__(config)
        self.config = config
        # Use Conv2D for weight storage (compatible with HF checkpoints),
        # but forward uses manual unfold+matmul for numerical precision alignment.
        self.patch_conv = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias_attr=False,
        )
        self.patch_size = config.patch_size
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.patch_conv

    def _patch_embed_forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        """Manual patch embedding using unfold+matmul for precise numerical alignment.

        Conv2D with kernel_size==stride is equivalent to:
        1. Unfold input into non-overlapping patches
        2. Matrix multiply with flattened kernel weights
        """
        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.patch_size
        h_patches = height // patch_size
        w_patches = width // patch_size

        # Reshape to patches: [B, C, H, W] -> [B, C, h_p, ps, w_p, ps] -> [B, h_p*w_p, C*ps*ps]
        x = pixel_values.reshape([batch_size, channels, h_patches, patch_size, w_patches, patch_size])
        x = x.transpose([0, 2, 4, 1, 3, 5])  # [B, h_p, w_p, C, ps, ps]
        x = x.reshape([batch_size, h_patches * w_patches, channels * patch_size * patch_size])

        # weight: [out_channels, in_channels, ps, ps] -> [out_channels, in_channels*ps*ps]
        weight = self.patch_conv.weight.reshape([self.config.hidden_size, -1])

        # matmul: [B, num_patches, C*ps*ps] @ [C*ps*ps, hidden_size] -> [B, num_patches, hidden_size]
        output = paddle.matmul(x, weight.transpose([1, 0]))

        # Reshape back to [B, hidden_size, h_patches, w_patches] for compatibility with the rest
        output = output.transpose([0, 2, 1]).reshape([batch_size, self.config.hidden_size, h_patches, w_patches])
        return output

    def forward(
        self,
        pixel_values: paddle.Tensor,
        image_sizes: paddle.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
            image_sizes: list of (height, width) tuples for each image
            output_hidden_states: whether to return all hidden states
            return_dict: whether to return BaseModelOutput or tuple
        """
        if image_sizes is None:
            batch_size, _, height, width = pixel_values.shape
            image_sizes = [(height, width)] * batch_size

        # Pass images through patch embedding (manual unfold+matmul for precision)
        target_dtype = self.patch_conv.weight.dtype
        patch_embeds = self._patch_embed_forward(pixel_values.astype(target_dtype))
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # Flatten to a single sequence: [1, total_patches, hidden_size]
        patch_embeds = paddle.concat([p.flatten(1).transpose([1, 0]) for p in patch_embeds_list], axis=0).unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        # Compute 2D position ids
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )
        position_ids = position_ids.unsqueeze(0)

        # Get position embeddings (cos, sin)
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        # Generate block attention mask
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

        # Forward through transformer
        outputs = self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
        )

        if return_dict:
            return outputs
        return (outputs.last_hidden_state, outputs.hidden_states)


# =============================================================================
# Full Pixtral VLM (LLaVA-style: PixtralVision + MLP Projector + Mistral/Llama)
# =============================================================================


class PixtralMultiModalProjector(nn.Layer):
    """Simple MLP projector (LLaVA-style): Linear -> GELU -> Linear."""

    def __init__(self, config: PixtralConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias_attr=config.multimodal_projector_bias,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PixtralVLMPretrainedModel(PretrainedModel):
    """Pretrained model base for full Pixtral VLM."""

    config_class = PixtralConfig
    base_model_prefix = "model"

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2",
        "lm_head",
    ]


@register_base_model
class PixtralModel(PixtralVLMPretrainedModel):
    """Pixtral VLM composite model: PixtralVisionModel + LlamaModel + MLP Projector."""

    base_model_prefix = ""

    def __init__(self, config: PixtralConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = PixtralVisionModel(config.vision_config)
        self.language_model = LlamaModel(config.text_config)
        self.multi_modal_projector = PixtralMultiModalProjector(config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: paddle.Tensor,
        image_sizes: list | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
    ) -> list[paddle.Tensor]:
        """Get image features from vision encoder and project."""
        if vision_feature_layer is None:
            vision_feature_layer = self.config.vision_feature_layer
        if vision_feature_select_strategy is None:
            vision_feature_select_strategy = self.config.vision_feature_select_strategy

        image_outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )

        # Select feature from specified layer(s)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[idx] for idx in vision_feature_layer]
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = paddle.concat(hs_pool, axis=-1)

        # Project to text hidden size
        image_features = self.multi_modal_projector(selected_image_feature)

        # Split by image if image_sizes provided
        if image_sizes is not None:
            patch_size = self.vision_tower.patch_size
            split_sizes = []
            for size in image_sizes:
                h = size[0] // patch_size
                w = size[1] // patch_size
                split_sizes.append(h * w)
            image_features = paddle.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)

        return image_features

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        """Get mask for image placeholder tokens."""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                paddle.to_tensor(self.config.image_token_index, dtype="int64")
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return special_image_mask

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        pixel_values: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        image_sizes: list | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = paddle.concat(image_features, axis=0).astype(inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        return outputs


class PixtralForConditionalGeneration(PixtralVLMPretrainedModel):
    """Pixtral VLM with language modeling head (LLaVA-style)."""

    def __init__(self, config: PixtralConfig):
        super().__init__(config)
        self.config = config
        self.model = PixtralModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias_attr=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def get_image_features(self, pixel_values, image_sizes=None, **kwargs):
        return self.model.get_image_features(pixel_values, image_sizes=image_sizes, **kwargs)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        pixel_values: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        image_sizes: list | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_labels = shift_labels.reshape([-1])
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )
