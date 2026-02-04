import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2Attention,
    Qwen2PreTrainedModel,
    Qwen2Config,
    Qwen2RotaryEmbedding,
    Qwen2RMSNorm,
    Qwen2DecoderLayer,
    BaseModelOutputWithPast,
    DynamicCache,
    create_causal_mask,
    create_sliding_window_causal_mask,
    check_model_inputs,
    Unpack,
    TransformersKwargs,
    apply_rotary_pos_emb,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, Cache
from typing import Optional
from cutile.ops.matmul_silu_mul import launch_matmul_silu_mul, launch_gemv_silu_mul
from cutile.ops.matmul import launch_matmul, launch_gemv
from cutile.ops.attention import fmha


def my_qwen2_mlp(
    stream: torch.cuda.Stream,
    self: Qwen2MLP,
    x: torch.Tensor,
    intermediate_buffer: torch.Tensor,
    output_buffer: torch.Tensor
) -> torch.Tensor:
    batch_size = x.size(0)
    xv = x.view(-1, x.size(-1))
    M = xv.size(0)
    # v0 = torch.empty((M, self.intermediate_size), device=x.device, dtype=torch.float16)
    if M == 1:
        launch_gemv_silu_mul(
            stream, xv, self.gate_proj.weight, self.up_proj.weight, intermediate_buffer, approx=True
        )
        launch_gemv(stream, intermediate_buffer, self.down_proj.weight, output_buffer)
    else:
        launch_matmul_silu_mul(
            stream, xv, self.gate_proj.weight, self.up_proj.weight, intermediate_buffer, approx=True
        )
        launch_matmul(stream, intermediate_buffer, self.down_proj.weight, output_buffer, transb=True, act=0)
        # a = torch.matmul(xv, self.gate_proj.weight.T)
        # v0 = torch.sigmoid(a) * a  * torch.matmul(xv, self.up_proj.weight.T)
        # torch.matmul(intermediate_buffer, self.down_proj.weight.T, out=output_buffer)
    return output_buffer.view(batch_size, -1, self.hidden_size)


def my_qwen2_self_attn(
    stream: torch.cuda.Stream,
    out: torch.Tensor,
    self: Qwen2Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_output, attn_weights = fmha(
        stream,
        out,
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def my_qwen2_decoder_layer(
    stream: torch.cuda.Stream,
    out: torch.Tensor,
    mlp_intermediate_buffer: torch.Tensor,
    mlp_output_buffer: torch.Tensor,
    self: Qwen2DecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _ = my_qwen2_self_attn(
        stream,
        out,
        self.self_attn,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = my_qwen2_mlp(stream, self.mlp, hidden_states, mlp_intermediate_buffer, mlp_output_buffer)
    hidden_states = residual + hidden_states
    return hidden_states


class MyQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        assert config.hidden_act in ["silu"], "Unsupported activation function"
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        stream = torch.cuda.current_stream()
        bs, seq_len, hidden_size = hidden_states.shape
        attn_out = torch.empty((bs, seq_len, self.config.num_attention_heads, hidden_size // self.config.num_attention_heads), device=hidden_states.device, dtype=hidden_states.dtype)
        mlp_intermediate_buffer = torch.empty((bs * seq_len, self.config.intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
        mlp_output_buffer = torch.empty((bs * seq_len, self.config.hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = my_qwen2_decoder_layer(
                stream,
                attn_out,
                mlp_intermediate_buffer,
                mlp_output_buffer,
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


if __name__ == "__main__":
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

    # Simple test
    class Config:
        hidden_size = 128
        intermediate_size = 64
        hidden_act = "silu"

    config = Config()
    x = torch.rand((8, 16, 128), device="cuda", dtype=torch.float16)
    mlp2 = qwen2_mod.Qwen2MLP(config).cuda().to(torch.float16)
    output = my_qwen2_mlp(torch.cuda.current_stream(), mlp2, x)
    output2 = mlp2(x)
    torch.testing.assert_close(output, output2, rtol=1e-5, atol=1e-4)
    print("✓ MyQwen2MLP test passed!")
