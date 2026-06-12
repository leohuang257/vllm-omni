# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.

With Option B enhancement, extractors now return a CacheContext object containing
all model-specific information needed for generic caching, including preprocessing,
transformer execution, and postprocessing logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    This allows the TeaCacheHook to remain completely generic - all model-specific
    logic is encapsulated in the extractor that returns this context.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Must be a torch.Tensor extracted from the first transformer block,
            typically after applying normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Must be a torch.Tensor representing the main image/latent states
            after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Must be a torch.Tensor containing the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]

            Returns:
                tuple containing:
                - [0]: processed hidden_states (required)
                - [1]: processed encoder_hidden_states (optional, only for dual-stream)

            Example for single-stream:
                def run_blocks():
                    h = hidden_states
                    for block in module.transformer_blocks:
                        h = block(h, temb=temb)
                    return (h,)

            Example for dual-stream:
                def run_blocks():
                    h, e = hidden_states, encoder_hidden_states
                    for block in module.transformer_blocks:
                        e, h = block(h, e, temb=temb)
                    return (h, e)

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]

            Takes the processed hidden_states and applies final transformations
            (normalization, projection) to produce the model output.

            Example:
                def postprocess(h):
                    h = module.norm_out(h, temb)
                    output = module.proj_out(h)
                    return Transformer2DModelOutput(sample=output)

        extra_states: Optional dict for additional model-specific state.
            Use this for models that need to pass additional context beyond
            the standard fields.
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, ...]]
    postprocess: Callable[[torch.Tensor], Any]
    extra_states: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate that the CacheContext contains valid data.

        Raises:
            TypeError: If fields have wrong types
            ValueError: If tensors have invalid properties
            RuntimeError: If callables fail basic invocation tests

        This method should be called after creating a CacheContext to catch
        common developer errors early with clear error messages.
        """
        # Validate tensor fields
        if not isinstance(self.modulated_input, torch.Tensor):
            raise TypeError(f"modulated_input must be torch.Tensor, got {type(self.modulated_input)}")

        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be torch.Tensor, got {type(self.hidden_states)}")

        if self.encoder_hidden_states is not None and not isinstance(self.encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"encoder_hidden_states must be torch.Tensor or None, got {type(self.encoder_hidden_states)}"
            )

        if not isinstance(self.temb, torch.Tensor):
            raise TypeError(f"temb must be torch.Tensor, got {type(self.temb)}")

        # Validate callables
        if not callable(self.run_transformer_blocks):
            raise TypeError(f"run_transformer_blocks must be callable, got {type(self.run_transformer_blocks)}")

        if not callable(self.postprocess):
            raise TypeError(f"postprocess must be callable, got {type(self.postprocess)}")

        # Validate tensor shapes are compatible
        if self.modulated_input.shape[0] != self.hidden_states.shape[0]:
            raise ValueError(
                f"Batch size mismatch: modulated_input has batch size "
                f"{self.modulated_input.shape[0]}, but hidden_states has "
                f"{self.hidden_states.shape[0]}"
            )

        # Validate devices match
        if self.modulated_input.device != self.hidden_states.device:
            raise ValueError(
                f"Device mismatch: modulated_input on {self.modulated_input.device}, "
                f"hidden_states on {self.hidden_states.device}"
            )


def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor | float | int,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    additional_t_cond: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for QwenImageTransformer2DModel.

    This is the ONLY Qwen-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: QwenImageTransformer2DModel instance
        hidden_states: Input hidden states tensor
        encoder_hidden_states: Text encoder outputs
        encoder_hidden_states_mask: Mask for text encoder
        timestep: Current diffusion timestep
        img_shapes: Image shapes for position embedding
        txt_seq_lens: Text sequence lengths
        guidance: Optional guidance scale for CFG
        additional_t_cond: Optional additional timestep conditioning
        attention_kwargs: Additional attention arguments
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Qwen-specific)
    # ============================================================================
    # Call image_rope_prepare instead of img_in + pos_embed directly.
    # This ensures the SequenceParallelSplitHook registered on image_rope_prepare
    # fires when SP is enabled, correctly sharding hidden_states and vid_freqs.
    hidden_states, vid_freqs, txt_freqs = module.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
    image_rotary_emb = (vid_freqs, txt_freqs)

    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)

    # Call modulate_index_prepare instead of handling timestep directly.
    # For zero_cond_t=False: timestep unchanged, modulate_index=None.
    # For zero_cond_t=True: timestep is doubled, modulate_index is created and
    # sharded by the SequenceParallelSplitHook on modulate_index_prepare so that
    # its sequence dimension matches the already-sharded hidden_states.
    timestep, modulate_index = module.modulate_index_prepare(timestep, img_shapes)

    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        module.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_scale1, img_shift1, _ = block._modulate(img_mod1)
    img_modulated = block.img_norm1(hidden_states, img_scale1, img_shift1)

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Qwen-specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Qwen transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states
        encoder_mask = encoder_hidden_states_mask
        hidden_states_mask = None  # default
        if module.parallel_config is not None and module.parallel_config.sequence_parallel_size > 1:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                # Create mask for the full (padded) sequence
                # valid positions = True, padding positions = False
                batch_size = hidden_states.shape[0]
                padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    batch_size,
                    padded_seq_len,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                hidden_states_mask[:, ctx.sp_original_seq_len :] = False

        # if mask is all true, set it to None
        if hidden_states_mask is not None and hidden_states_mask.all():
            hidden_states_mask = None
        if encoder_mask is not None and encoder_mask.all():
            encoder_mask = None
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
                hidden_states_mask=hidden_states_mask,
            )
        return (h, e)

    # ============================================================================
    # DEFINE POSTPROCESSING (Qwen-specific)
    # ============================================================================
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        """Apply Qwen-specific output postprocessing."""
        if getattr(module, "zero_cond_t", False):
            h = module.norm_out(h, temb.chunk(2, dim=0)[0])
        else:
            h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_bagel_context(
    module: nn.Module,
    x_t: torch.Tensor,
    timestep: torch.Tensor | float | int,
    packed_vae_token_indexes: torch.LongTensor,
    packed_vae_position_ids: torch.LongTensor,
    packed_text_ids: torch.LongTensor,
    packed_text_indexes: torch.LongTensor,
    packed_position_ids: torch.LongTensor,
    packed_seqlens: torch.IntTensor,
    past_key_values: Any,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Bagel model.

    Args:
        module: Bagel instance
        x_t: Latent image input
        timestep: Current timestep
        packed_vae_token_indexes: Indexes for VAE tokens in packed sequence
        packed_vae_position_ids: Position IDs for VAE tokens
        packed_text_ids: Text token IDs
        packed_text_indexes: Indexes for text tokens in packed sequence
        packed_position_ids: Global position IDs
        packed_seqlens: Sequence lengths
        past_key_values: KV cache
        **kwargs: Additional keyword arguments

    Returns:
        CacheContext with all information needed for generic caching
    """

    # 1. Embed text
    packed_text_embedding = module.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), module.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    # 2. Embed timestep
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=x_t.device)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    # 3. Embed image (x_t)
    packed_pos_embed = module.latent_pos_embed(packed_vae_position_ids)
    packed_timestep_embeds = module.time_embedder(timestep)

    x_t_emb = module.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
    if x_t_emb.dtype != packed_sequence.dtype:
        x_t_emb = x_t_emb.to(packed_sequence.dtype)

    packed_sequence[packed_vae_token_indexes] = x_t_emb

    # Use the full packed sequence as modulated input to match hidden_states size
    modulated_input = packed_sequence

    def run_transformer_blocks():
        extra_inputs = {}
        if module.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = module.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        return (output.packed_query_sequence,)

    def postprocess(h):
        v_t = module.llm2vae(h)
        v_t = v_t[packed_vae_token_indexes]
        return v_t

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=packed_sequence,  # Use full packed sequence
        encoder_hidden_states=None,
        temb=packed_timestep_embeds,  # Approximate
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_zimage_context(
    module: nn.Module,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    patch_size: int = 2,
    f_patch_size: int = 1,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for ZImageTransformer2DModel.

    This is the ONLY Z-Image-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: ZImageTransformer2DModel instance
        x: List of image tensors per batch item
        t: Timestep tensor
        cap_feats: List of caption feature tensors per batch item
        patch_size: Patch size for patchification (default: 2)
        f_patch_size: Frame patch size (default: 1)
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from torch.nn.utils.rnn import pad_sequence

    if not hasattr(module, "layers") or len(module.layers) == 0:
        raise ValueError("Module must have main transformer layers")

    bsz = len(x)
    device = x[0].device

    # ============================================================================
    # PREPROCESSING (Z-Image specific)
    # ============================================================================
    # Scale timestep and create timestep embedding
    t_scaled = t * module.t_scale
    adaln_input = module.t_embedder(t_scaled)

    # Patchify and embed inputs
    (
        x_patches,
        cap_feats_processed,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = module.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # Process image patches through embedder and noise refiner
    x_item_seqlens = [len(_) for _ in x_patches]
    x_max_item_seqlen = max(x_item_seqlens)

    x_embedded = torch.cat(x_patches, dim=0)
    x_embedded = module.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_embedded)

    # Match adaln_input dtype to x_embedded
    adaln_input = adaln_input.type_as(x_embedded)

    # Apply pad token
    x_embedded[torch.cat(x_inner_pad_mask)] = module.x_pad_token
    x_list = list(x_embedded.split(x_item_seqlens, dim=0))

    # Compute rope embeddings for image patches
    x_cos, x_sin = module.rope_embedder(torch.cat(x_pos_ids, dim=0))
    x_cos = list(x_cos.split(x_item_seqlens, dim=0))
    x_sin = list(x_sin.split(x_item_seqlens, dim=0))

    # Pad sequences for batch processing
    x_batched = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    x_cos_batched = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
    x_sin_batched = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    # Run noise refiner blocks
    for layer in module.noise_refiner:
        x_batched = layer(x_batched, x_attn_mask, x_cos_batched, x_sin_batched, adaln_input)

    # Process caption features through embedder and context refiner
    cap_item_seqlens = [len(_) for _ in cap_feats_processed]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_embedded = torch.cat(cap_feats_processed, dim=0)
    cap_embedded = module.cap_embedder(cap_embedded)
    cap_embedded[torch.cat(cap_inner_pad_mask)] = module.cap_pad_token
    cap_list = list(cap_embedded.split(cap_item_seqlens, dim=0))

    # Compute rope embeddings for caption
    cap_cos, cap_sin = module.rope_embedder(torch.cat(cap_pos_ids, dim=0))
    cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
    cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))

    # Pad sequences for batch processing
    cap_batched = pad_sequence(cap_list, batch_first=True, padding_value=0.0)
    cap_cos_batched = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
    cap_sin_batched = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    # Run context refiner blocks
    for layer in module.context_refiner:
        cap_batched = layer(cap_batched, cap_attn_mask, cap_cos_batched, cap_sin_batched)

    # Create unified sequence (image + caption)
    unified_list = []
    unified_cos_list = []
    unified_sin_list = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified_list.append(torch.cat([x_batched[i][:x_len], cap_batched[i][:cap_len]]))
        unified_cos_list.append(torch.cat([x_cos_batched[i][:x_len], cap_cos_batched[i][:cap_len]]))
        unified_sin_list.append(torch.cat([x_sin_batched[i][:x_len], cap_sin_batched[i][:cap_len]]))

    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
    unified_cos = pad_sequence(unified_cos_list, batch_first=True, padding_value=0.0)
    unified_sin = pad_sequence(unified_sin_list, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    # Use the first main transformer block's modulation
    # The main layers have modulation=True and process the unified sequence
    block = module.layers[0]
    # Get modulation parameters: scale_msa, gate_msa, scale_mlp, gate_mlp
    mod_params = block.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
    scale_msa = 1.0 + mod_params[0]
    # Extract modulated input: normalized hidden states scaled by modulation
    modulated_input = block.attention_norm1(unified) * scale_msa

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Z-Image specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Z-Image main transformer blocks."""
        h = unified
        for layer in module.layers:
            h = layer(h, unified_attn_mask, unified_cos, unified_sin, adaln_input)
        return (h,)

    # ============================================================================
    # DEFINE POSTPROCESSING (Z-Image specific)
    # ============================================================================
    def postprocess(h):
        """Apply Z-Image specific output postprocessing."""
        h = module.all_final_layer[f"{patch_size}-{f_patch_size}"](h, adaln_input)
        h = list(h.unbind(dim=0))
        output = module.unpatchify(h, x_size, patch_size, f_patch_size)
        return output, {}

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=unified,
        encoder_hidden_states=None,  # Z-Image uses unified sequence, no separate encoder states
        temb=adaln_input,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
        extra_states={
            "unified_attn_mask": unified_attn_mask,
            "unified_cos": unified_cos,
            "unified_sin": unified_sin,
            "x_size": x_size,
            "x_item_seqlens": x_item_seqlens,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
        },
    )


def extract_flux2_klein_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Flux2Klein model.

    Caches the full transformer output (including single_transformer_blocks).
    When cache is reused, single_transformer_blocks is skipped to achieve maximum speedup.

    Args:
        module: Flux2Transformer2DModel instance
        hidden_states: Input image hidden states tensor
        encoder_hidden_states: Input text hidden states tensor
        timestep: Current diffusion timestep
        img_ids: Image position IDs for RoPE
        txt_ids: Text position IDs for RoPE
        guidance: Optional guidance scale for CFG
        joint_attention_kwargs: Additional attention kwargs

    Returns:
        CacheContext with all information needed for generic caching
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Flux2-specific)
    # ============================================================================
    dtype = hidden_states.dtype

    num_txt_tokens = encoder_hidden_states.shape[1]

    timestep = timestep.to(dtype=dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(dtype=dtype) * 1000

    temb = module.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = module.double_stream_modulation_img(temb)
    double_stream_mod_txt = module.double_stream_modulation_txt(temb)
    single_stream_mod = module.single_stream_modulation(temb)[0]

    hidden_states = module.x_embedder(hidden_states)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = module.pos_embed(img_ids)
    text_rotary_emb = module.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    block = module.transformer_blocks[0]

    norm_hidden_states = block.norm1(hidden_states)
    norm_hidden_states = (1 + double_stream_mod_img[0][1]) * norm_hidden_states + double_stream_mod_img[0][0]

    modulated_input = norm_hidden_states

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Flux2-specific)
    # ============================================================================
    def run_flux2_transformer_blocks():
        h = hidden_states
        c = encoder_hidden_states
        for block in module.transformer_blocks:
            c, h = block(
                hidden_states=h,
                encoder_hidden_states=c,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        return (h, c)

    def run_flux2_full_transformer_with_single(ori_h, ori_c):
        h = ori_h
        c = ori_c
        for block in module.transformer_blocks:
            c, h = block(
                hidden_states=h,
                encoder_hidden_states=c,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        h_concat = torch.cat([c, h], dim=1)
        for block in module.single_transformer_blocks:
            h_concat = block(
                hidden_states=h_concat,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        final_hidden_states = h_concat[:, num_txt_tokens:, ...]
        return final_hidden_states, c

    # ============================================================================
    # DEFINE POSTPROCESSING (Flux2-specific)
    # ============================================================================
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        h = module.norm_out(h, temb)
        h = module.proj_out(h)
        if not return_dict:
            return (h,)
        return Transformer2DModelOutput(sample=h)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_flux2_transformer_blocks,
        postprocess=postprocess,
        extra_states={
            "run_flux2_full_transformer_with_single": run_flux2_full_transformer_with_single,
        },
    )


def extract_longcat_context(
    module: nn.Module,  # LongCatImageTransformer2DModel
    hidden_states,
    timestep,
    guidance,
    encoder_hidden_states,
    txt_ids,
    img_ids,
    **kwargs,
) -> CacheContext:
    """Extract the cache context for LongCat Image.

    Similar to other extractors, this is currently the only code needed
    for TeaCache support for LongCat image, and encapsulates preprocessing,
    modulated input extraction, transformer execution, and postprocessing
    logic.

    Args & kawrgs are identical to the inputs to LongCat Image's forward.

    Returns:
        CacheContext with all information needed for generic caching
    """
    # TODO (Alex) - Refactor TeaCache extractors to more tightly integrate with .forward
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    # 1. Model specific preprocessing
    fwd_context = get_forward_context()
    sp_size = module.parallel_config.sequence_parallel_size
    if sp_size is not None and sp_size > 1:
        # NOTE: For now, we set this to False on the forward context
        # to be consistent with LongCat Image's current behavior when
        # TeaCache is enabled. We do not need to reset it in post process
        # since we should never split text embed in sp for this model.
        fwd_context.split_text_embed_in_sp = False

    hidden_states = module.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000

    temb = module.time_embed(timestep, hidden_states.dtype)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    # Compute RoPE embeddings via rope_preparer module
    # _sp_plan will automatically shard img_cos/img_sin (outputs 2, 3)
    # txt_cos/txt_sin (outputs 0, 1) remain replicated for dual-stream attention
    txt_cos, txt_sin, img_cos, img_sin = module.rope_preparer(txt_ids, img_ids)

    # Reconstruct image_rotary_emb with chunked values
    # Final shape: (txt_seq_len + img_seq_len // SP, head_dim)
    image_rotary_emb = (
        torch.cat([txt_cos, img_cos], dim=0),
        torch.cat([txt_sin, img_sin], dim=0),
    )

    # 2. Extract the modulated output from the first mm-DiT block
    first_block = module.transformer_blocks[0]
    img_modulated = first_block.norm1(hidden_states, emb=temb)[0]

    # 3. Define the transformer execution
    def run_transformer_blocks():
        """Execute all Longcat transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for block in module.single_transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        # Hook expects hidden states to be first
        return (h, e)

    # 4. Postprocessing
    def postprocess(h):
        """Apply Longcat-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        return Transformer2DModelOutput(sample=output)

    # 5. Return the CacheContext
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_stable_audio_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    global_hidden_states: torch.Tensor | None = None,
    rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_dict: bool = True,
    attention_mask: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Stable Audio DiT model.

    Architecture Notes:
        - Stable Audio uses standard LayerNorm
        - Timestep conditioning via global_hidden_states prepended to sequence
        - Single-stream model (cross-attention handled separately)
        - Input: [B, C, L] (C=in_channels) -> transpose -> [B, L, C] -> project -> [B, L, inner_dim]
        - Global states prepended: [B, 1+L, inner_dim]
    """
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks attribute with at least one block")

    # Cast inputs to match model weights dtype
    hidden_states = hidden_states.to(module.dtype)
    encoder_hidden_states = encoder_hidden_states.to(module.dtype)
    if global_hidden_states is not None:
        global_hidden_states = global_hidden_states.to(module.dtype)

    cross_attention_hidden_states = module.cross_attention_proj(encoder_hidden_states)
    global_hidden_states = module.global_proj(global_hidden_states)
    time_hidden_states = module.timestep_proj(module.time_proj(timestep.to(module.dtype)))

    # Combine global and time embeddings → [B, 1, inner_dim]
    temb = global_hidden_states + time_hidden_states.unsqueeze(1)

    hidden_states = module.preprocess_conv(hidden_states) + hidden_states
    hidden_states = hidden_states.transpose(1, 2)

    # Capture the original sequence length for safe post-processing slicing
    original_seq_len = hidden_states.shape[1]

    hidden_states = module.proj_in(hidden_states)
    hidden_states = torch.cat([temb, hidden_states], dim=1)

    # attention_mask is not used in the standard Stable Audio Open inference path.
    assert attention_mask is None, (
        "attention_mask is not supported in extract_stable_audio_context; expected None for Stable Audio Open."
    )

    # Stable Audio prepends the combined global+time embedding (`temb`) to the sequence.
    # Therefore, the standard LayerNorm applied here still captures the timestep signal
    # within the first token of the output, giving the cache discriminator the info it needs.
    first_block = module.transformer_blocks[0]
    modulated_input = first_block.norm1(hidden_states)

    def run_transformer_blocks() -> tuple[torch.Tensor]:
        """
        Execute all Stable Audio transformer blocks.

        Returns:
            Tuple containing only hidden_states (single-stream model).
            Format: (hidden_states,)
        """
        h = hidden_states
        for block in module.transformer_blocks:
            h = block(
                hidden_states=h,
                encoder_hidden_states=cross_attention_hidden_states,
                rotary_embedding=rotary_embedding,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        return (h,)

    def postprocess(h: torch.Tensor) -> Any:
        """
        Apply Stable Audio-specific output postprocessing.

        Args:
            h: Hidden states from transformer blocks [B, 1+L, inner_dim]

        Returns:
            Transformer2DModelOutput or tuple based on return_dict
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        h = module.proj_out(h)
        h = h.transpose(1, 2)[:, :, -original_seq_len:]
        output = module.postprocess_conv(h) + h
        if return_dict:
            return Transformer2DModelOutput(sample=output)
        return (output,)

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=None,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_flux2_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Flux2Transformer2DModel.

    This is the ONLY Flux2-specific code needed for TeaCache support.
    It encapsulates preprocessing, modulated input extraction, transformer execution,
    and postprocessing logic.

    Args:
        module: Flux2Transformer2DModel instance
        hidden_states: Input hidden states tensor
        encoder_hidden_states: Text encoder outputs
        timestep: Current diffusion timestep
        img_ids: Image inputs for position embedding
        txt_ids: Text inputs for position embedding
        guidance: Optional guidance scale for CFG
        joint_attention_kwargs: Additional attention arguments
        return_dict: Whether to return a Transformer2DModelOutput instead of a plain tensor
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """

    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Flux2-specific)
    # ============================================================================
    num_txt_tokens = encoder_hidden_states.shape[1]

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = module.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = module.double_stream_modulation_img(temb)
    double_stream_mod_txt = module.double_stream_modulation_txt(temb)
    single_stream_mod = module.single_stream_modulation(temb)[0]

    hidden_states = module.x_embedder(hidden_states)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    if current_omni_platform.is_npu():
        freqs_cos_image, freqs_sin_image = module.pos_embed(img_ids.cpu())
        image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
        freqs_cos_text, freqs_sin_text = module.pos_embed(txt_ids.cpu())
        text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
    else:
        image_rotary_emb = module.pos_embed(img_ids)
        text_rotary_emb = module.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    block = module.transformer_blocks[0]
    (shift_msa, scale_msa, gate_msa), _ = double_stream_mod_img
    modulated_input = block.norm1(hidden_states)
    modulated_input = (1 + scale_msa) * modulated_input + shift_msa

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Flux2-specific)
    # ============================================================================
    def run_transformer_blocks():
        """Execute all Flux2 transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states

        for transformer_block in module.transformer_blocks:
            e, h = transformer_block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        h = torch.cat([e, h], dim=1)

        for single_transformer_block in module.single_transformer_blocks:
            h = single_transformer_block(
                hidden_states=h,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        h = h[:, num_txt_tokens:, ...]
        return (h,)

    # ============================================================================
    # DEFINE POSTPROCESSING
    # ============================================================================
    def postprocess(h):
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_flux_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.LongTensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for FluxTransformer2DModel.

    This mirrors the standard FLUX.1 transformer forward path while exposing
    the first modulated image stream tensor as TeaCache's similarity signal.

    Args:
        module: FluxTransformer2DModel instance
        hidden_states: Input image hidden states tensor
        encoder_hidden_states: Text encoder outputs
        pooled_projections: Pooled text conditioning
        timestep: Current diffusion timestep
        img_ids: Image position IDs for RoPE
        txt_ids: Text position IDs for RoPE
        guidance: Optional guidance scale for guidance-distilled variants
        joint_attention_kwargs: Additional attention kwargs
        **kwargs: Additional keyword arguments

    Returns:
        CacheContext with all information needed for generic caching
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # ============================================================================
    # PREPROCESSING (Flux1-specific)
    # ============================================================================
    hidden_states = module.x_embedder(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

    temb = (
        module.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else module.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    if current_omni_platform.is_npu():
        freqs_cos, freqs_sin = module.pos_embed(ids.cpu())
        image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
    else:
        image_rotary_emb = module.pos_embed(ids)

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    first_block = module.transformer_blocks[0]
    modulated_input, *_ = first_block.norm1(hidden_states, emb=temb)

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Flux1-specific)
    # ============================================================================
    def run_transformer_blocks() -> tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states
        e = encoder_hidden_states

        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for block in module.single_transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        return (h, e)

    return_dict = kwargs.get("return_dict", True)

    # ============================================================================
    # DEFINE POSTPROCESSING
    # ============================================================================
    def postprocess(h: torch.Tensor) -> Any:
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_sensenova_u1_context(
    module: nn.Module,
    input_ids: torch.Tensor | None = None,
    indexes: torch.Tensor | None = None,
    attention_mask: dict[str, torch.Tensor] | torch.Tensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.Tensor | None = None,
    use_cache: bool | None = None,
    embed_only: bool = False,
    compute_logits: bool = True,
    **kwargs: Any,
) -> CacheContext:
    """Extract cache context for SenseNovaU1ForCausalLM denoising forwards."""
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
        SenseNovaU1CausalLMOutput,
    )

    layer_kwargs = dict(kwargs)
    layer_kwargs.pop("cache_dit_skip", None)
    image_gen_indicators = layer_kwargs.pop("image_gen_indicators", None)
    exist_und = (~image_gen_indicators).any().item()
    exist_gen = image_gen_indicators.any().item()
    causal_mask_mapping = attention_mask

    first_layer = module.model.layers[0]
    modulated_input = first_layer.input_layernorm_mot_gen(inputs_embeds)

    def run_transformer_blocks():
        h = inputs_embeds
        for layer in module.model.layers:
            h = layer(
                h,
                image_gen_indicators=image_gen_indicators,
                exist_und=exist_und,
                exist_gen=exist_gen,
                indexes=indexes,
                attention_mask=causal_mask_mapping,
                past_key_values=past_key_values,
                **layer_kwargs,
            )
        return (h,)

    def postprocess(h: torch.Tensor) -> SenseNovaU1CausalLMOutput:
        h = module.model.norm_mot_gen(h)

        logits = module.logits_processor(module.lm_head, h) if compute_logits else None
        return SenseNovaU1CausalLMOutput(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=h,
        )

    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=inputs_embeds,
        encoder_hidden_states=None,
        temb=modulated_input,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


def extract_cosmos3_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    video_shape: tuple[int, int, int],
    fps: float | None = None,
    action_latents: torch.Tensor | None = None,
    action_domain_ids: torch.Tensor | None = None,
    action_noisy_mask: torch.Tensor | None = None,
    action_start_frame_offset: int = 1,
    action_fps: float | None = None,
    sound_latents: torch.Tensor | None = None,
    noisy_frame_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    """
    Extract cache context for Cosmos3VFMTransformer.

    This mirrors the Cosmos3VFMTransformer GEN forward path while exposing
    the first GEN layer's input RMSNorm output as TeaCache's similarity signal.

    Architecture Notes:
        - MoT model: UND language model runs once per generation (K/V cached on
          the module by the pipeline); GEN layers run every denoising step.
          TeaCache caches only the GEN pathway.
        - No adaLN: timestep embedding is added directly to GEN tokens; the first
          GEN layer's input RMSNorm output is the cache similarity signal.
        - Video, action, and sound tokens are concatenated into one GEN sequence;
          the cached residual covers the full sequence.

    Args:
        module: Cosmos3VFMTransformer instance
        hidden_states: Noisy video latents [B, C, t, h, w]
        timestep: Diffusion timestep [B]
        text_ids: Tokenized text [B, S_text]
        text_mask: Text attention mask [B, S_text]
        video_shape: Latent-space shape (t, h, w)
        fps: Video frame rate for temporal RoPE
        action_latents: Optional noisy action latents
        action_domain_ids: Optional embodiment domain IDs
        action_noisy_mask: Optional per-action-token noisy mask
        action_start_frame_offset: Action RoPE frame offset
        action_fps: Action frame rate for RoPE
        sound_latents: Optional noisy sound latents
        noisy_frame_mask: Optional per-frame noisy mask for I2V/V2V
        **kwargs: Additional keyword arguments ignored by this extractor

    Returns:
        CacheContext with all information needed for generic caching
    """
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import _get_ulysses_state

    if not hasattr(module, "gen_layers") or len(module.gen_layers) == 0:
        raise ValueError("Module must have gen_layers")

    # ============================================================================
    # PREPROCESSING (Cosmos3-specific)
    # ============================================================================
    t, h, w = video_shape
    hp, wp, _, _ = module._pad_to_patch_size(h, w)
    text_lengths = text_mask.sum(dim=1)
    min_real_len = int(text_lengths.min().item())
    max_real_len = int(text_lengths.max().item())
    if min_real_len != max_real_len:
        raise ValueError(
            f"Cosmos3 requires identical real text lengths within a batch (got min={min_real_len}, max={max_real_len})."
        )
    has_action = action_latents is not None
    has_sound = sound_latents is not None
    if has_action and not module.action_gen:
        raise ValueError(
            "Cosmos3 action generation was requested, but this transformer "
            "was initialized without action modules. Check that the "
            "transformer config enables action_gen."
        )
    if has_sound and not module.sound_gen:
        raise ValueError(
            "Cosmos3 sound generation was requested, but this transformer "
            "was initialized without sound modules. Check that the "
            "transformer config enables sound_gen or defines sound_dim."
        )

    ulysses_size, _, _ = _get_ulysses_state()

    # Patchify latents and project to hidden space
    hidden_video = module.proj_in(module.patchify(hidden_states, t, h, w))
    s_video = hidden_video.shape[1]
    s_action = 0
    hidden_action = None
    s_sound = 0
    hidden_sound = None
    if action_latents is not None:
        if action_latents.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "Cosmos3 action and video batch sizes must match: "
                f"video={hidden_states.shape[0]}, action={action_latents.shape[0]}."
            )
        if action_domain_ids is None:
            action_domain_ids = torch.zeros(action_latents.shape[0], dtype=torch.long, device=action_latents.device)
        hidden_action = module.action_proj_in(module.pack_action(action_latents), action_domain_ids)
        hidden_action = hidden_action + module.action_modality_embed.to(hidden_action.dtype)
        s_action = hidden_action.shape[1]
    if sound_latents is not None:
        if sound_latents.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "Cosmos3 sound and video batch sizes must match: "
                f"video={hidden_states.shape[0]}, sound={sound_latents.shape[0]}."
            )
        hidden_sound = module.audio_proj_in(module.pack_sound(sound_latents))
        hidden_sound = hidden_sound + module.audio_modality_embed.to(hidden_sound.dtype)
        s_sound = hidden_sound.shape[1]

    # Timestep embedding (fp32); added only to noisy tokens.
    with torch.autocast("cuda", enabled=True, dtype=torch.float32):
        time_embed = module.time_embedder(timestep * module.timestep_scale)
    time_embed = time_embed.to(hidden_states.dtype)

    if noisy_frame_mask is not None:
        token_noisy_mask = (
            noisy_frame_mask[:, 0, :, 0, 0].unsqueeze(-1).expand(-1, -1, hp * wp).reshape(hidden_video.shape[0], -1, 1)
        )
        hidden_video = hidden_video + time_embed.unsqueeze(1) * token_noisy_mask
    else:
        hidden_video = hidden_video + time_embed.unsqueeze(1)

    if hidden_action is not None:
        if action_noisy_mask is None:
            hidden_action = hidden_action + time_embed.unsqueeze(1)
        else:
            if action_noisy_mask.shape != (hidden_action.shape[0], hidden_action.shape[1], 1):
                raise ValueError(
                    f"Cosmos3 action_noisy_mask must have shape [B, T_action, 1], got {tuple(action_noisy_mask.shape)}."
                )
            hidden_action = hidden_action + time_embed.unsqueeze(1) * action_noisy_mask.to(hidden_action.dtype)

    if hidden_sound is not None:
        hidden_sound = hidden_sound + time_embed.unsqueeze(1)
    hidden_parts = [hidden_video]
    if hidden_action is not None:
        hidden_parts.append(hidden_action)
    if hidden_sound is not None:
        hidden_parts.append(hidden_sound)
    hidden_gen = torch.cat(hidden_parts, dim=1)

    # Run UND pathway once; pipeline swaps cached K/V between CFG branches.
    if module.cached_kv is None:
        freqs_und, freqs_gen_full = module._compute_rope_freqs(
            text_mask,
            t,
            hp,
            wp,
            fps,
            hidden_states.device,
            hidden_states.dtype,
            t_action=s_action,
            action_start_frame_offset=action_start_frame_offset,
            action_fps=action_fps,
            t_sound=s_sound,
        )
        cached_kv_full = module.language_model(text_ids, freqs_und)
        module.cached_freqs_gen = freqs_gen_full

        module.cached_kv = [(k[:, :max_real_len], v[:, :max_real_len]) for k, v in cached_kv_full]

    # Mirrors the matching guard in Cosmos3VFMTransformer.forward (kept for diff sync).
    if module.cached_kv is None or module.cached_freqs_gen is None:
        raise RuntimeError("Cosmos3 GEN cache was not initialized before running GEN layers.")
    module._validate_gen_sequence_parallel(
        s_gen=hidden_gen.shape[1],
        s_video=s_video,
        s_action=s_action,
        s_sound=s_sound,
        has_action=has_action,
        has_sound=has_sound,
        ulysses_size=ulysses_size,
    )
    # Call gen_sp_prepare so SequenceParallelInput hooks fire when SP is enabled.
    freqs_cos, freqs_sin = module.cached_freqs_gen
    hidden_gen, freqs_cos, freqs_sin = module.gen_sp_prepare(hidden_gen, freqs_cos, freqs_sin)
    cached_kv = module.cached_kv

    # ============================================================================
    # EXTRACT MODULATED INPUT (for cache decision)
    # ============================================================================
    # Cosmos3 has no adaLN modulation; the timestep embedding was added to
    # every noisy GEN token above, so the first GEN layer's input RMSNorm
    # output carries the timestep signal the cache discriminator needs.
    first_layer = module.gen_layers[0]
    modulated_input = first_layer.input_layernorm(hidden_gen)

    # ============================================================================
    # DEFINE TRANSFORMER EXECUTION (Cosmos3-specific)
    # ============================================================================
    def run_transformer_blocks() -> tuple[torch.Tensor]:
        """Execute all Cosmos3 GEN decoder layers.

        Returns:
            Tuple containing only hidden_states (single-stream GEN pathway).
            Format: (hidden_states,)
        """
        hidden = hidden_gen
        for layer, (k_und, v_und) in zip(module.gen_layers, cached_kv, strict=True):
            hidden = layer(
                hidden,
                k_und=k_und,
                v_und=v_und,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            )
        return (hidden,)

    # ============================================================================
    # DEFINE POSTPROCESSING
    # ============================================================================
    def postprocess(hidden: torch.Tensor) -> Any:
        """Apply Cosmos3-specific output postprocessing.

        Args:
            hidden: Hidden states from GEN layers [B, S_gen, hidden_size]

        Returns:
            Video velocity tensor, or tuple in video/action/sound order
        """
        hidden = module.gen_sp_gather(hidden)
        hidden = module.norm_moe_gen(hidden)
        if not has_action and not has_sound:
            return module.unpatchify(module.proj_out(hidden), t, h, w)

        split_sizes = [s_video]
        if has_action:
            split_sizes.append(s_action)
        if has_sound:
            split_sizes.append(s_sound)
        split_hidden = hidden.split(split_sizes, dim=1)
        video_pred = module.unpatchify(module.proj_out(split_hidden[0]), t, h, w)
        outputs: list[torch.Tensor] = [video_pred]
        split_idx = 1
        if has_action:
            assert action_domain_ids is not None
            outputs.append(module.unpack_action(module.action_proj_out(split_hidden[split_idx], action_domain_ids)))
            split_idx += 1
        if has_sound:
            outputs.append(module.unpack_sound(module.audio_proj_out(split_hidden[split_idx])))
        return tuple(outputs)

    # ============================================================================
    # RETURN CONTEXT
    # ============================================================================
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_gen,
        encoder_hidden_states=None,  # UND text conditioning enters via cached K/V
        temb=time_embed,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )


# Registry for model-specific extractors
# Key: Transformer class name
# Value: extractor function with signature (module, *args, **kwargs) -> CacheContext
#
# Note: Use the transformer class name as specified in pipelines as TeaCache hooks operate
# on the transformer module and multiple pipelines can share the same transformer.
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    "ZImageTransformer2DModel": extract_zimage_context,
    "Flux2Klein": extract_flux2_klein_context,
    "StableAudioDiTModel": extract_stable_audio_context,
    "Flux2Transformer2DModel": extract_flux2_context,
    "LongCatImageTransformer2DModel": extract_longcat_context,
    "FluxTransformer2DModel": extract_flux_context,
    "SenseNovaU1ForCausalLM": extract_sensenova_u1_context,
    "Cosmos3VFMTransformer": extract_cosmos3_context,
    # Future models:
    # "CogVideoXTransformer3DModel": extract_cogvideox_context,
}


def register_extractor(transformer_cls_name: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
        extractor_fn: Function with signature (module, *args, **kwargs) -> CacheContext

    Example:
        >>> def extract_flux_context(module, hidden_states, timestep, guidance=None, **kwargs):
        ...     # Preprocessing
        ...     temb = module.time_text_embed(timestep, guidance)
        ...     # Extract modulated input
        ...     modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)
        ...     # Define execution
        ...     def run_blocks():
        ...         h = hidden_states
        ...         for block in module.transformer_blocks:
        ...             h = block(h, temb=temb)
        ...         return (h,)
        ...     # Define postprocessing
        ...     def postprocess(h):
        ...         return module.proj_out(module.norm_out(h, temb))
        ...     # Return context
        ...     return CacheContext(modulated, hidden_states, None, temb, run_blocks, postprocess)
        >>> register_extractor("FluxTransformer2DModel", extract_flux_context)
    """
    EXTRACTOR_REGISTRY[transformer_cls_name] = extractor_fn


def get_extractor(transformer_cls_name: str) -> Callable:
    """
    Get extractor function for given transformer class.

    This function looks up the extractor based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "QwenImageTransformer2DModel")
                                Must exactly match a key in EXTRACTOR_REGISTRY.

    Returns:
        Extractor function with signature (module, *args, **kwargs) -> CacheContext

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Get extractor for QwenImageTransformer2DModel
        >>> extractor = get_extractor("QwenImageTransformer2DModel")
        >>> ctx = extractor(transformer, hidden_states, encoder_hidden_states, timestep, ...)
    """
    # Direct lookup - no substring matching
    if transformer_cls_name in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[transformer_cls_name]

    # No match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: '{transformer_cls_name}'. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
