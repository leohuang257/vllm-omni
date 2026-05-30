# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor: LeLM (Stage 0) -> Flow1dVAE (Stage 1).

Converts [num_frames, K=3] codec tokens into stream-major flat layout
for the diffusion decoder. Supports both sync and async-chunk paths.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)

logger = init_logger(__name__)

_SAVE_CODES_PATH = os.environ.get("SONGGEN_SAVE_CODES")


# Token IDs from configuration_songgeneration_v2.py.
_CODE_DEPTH = 3
_VALID_CODE_MAX = 16383  # 0..16383 are valid RVQ indices
_EOS_TOKEN_ID = 16384  # per-stream EOS
_SPECIAL_TOKEN_ID = 16385  # pad / mask / null
_VOCAL_REPLACEMENT = 3142  # used when gen_type='bgm' (vocal-stream filler)
_BGM_REPLACEMENT = 9670  # used when gen_type='vocal' (bgm-stream filler)


# ---------------------------------------------------------------------------
# Sync path: full-LM-output -> single diffusion call
# ---------------------------------------------------------------------------


def lelm_to_flow1dvae(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Pack the LeLM's per-request [num_frames, 3] codes for the decoder.

    Following the Fish Speech `slow_ar_to_dac_decoder` contract:

    * Each `source_outputs[i]` is a finished Stage-0 RequestOutput.
    * `output.multimodal_output["audio_codes"]` is [num_frames, K=3]
      mixed/vocal/bgm codes (frame-major).
    * Pad rows are filtered (special_token=16385 from delay pattern).
    * Optional `gen_type` ("mixed" | "vocal" | "bgm") triggers pure-track
      replacement: the unused stream is filled with a constant token.

    Returns a list of OmniTokensPrompt whose `prompt_token_ids` is the
    stream-major flat layout [s0...s0, s1...s1, s2...s2].
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    flow1dvae_inputs: list[OmniTokensPrompt] = []

    for i, lelm_output in enumerate(source_outputs):
        if not getattr(lelm_output, "finished", True):
            continue

        out = lelm_output.outputs[0]
        mm = getattr(out, "multimodal_output", None) or {}

        codes_BKT = _extract_audio_codes(mm)
        if codes_BKT is None:
            logger.warning("lelm_to_flow1dvae: request %d had no audio_codes in multimodal_output", i)
            flow1dvae_inputs.append(_empty_omni_tokens_prompt())
            continue

        # codes_BKT shape: [num_frames, K=3]. Filter rows that are entirely
        # pad/EOS (no valid RVQ index in any stream) so the diffusion
        # decoder doesn't see stretched silence.
        valid_mask = _valid_frame_mask(codes_BKT)
        codes_BKT = codes_BKT[valid_mask]

        if codes_BKT.numel() == 0:
            flow1dvae_inputs.append(_empty_omni_tokens_prompt())
            continue

        # Pure-track replacement (gen_type='bgm'/'vocal').
        gen_type = _resolve_gen_type(prompt, lelm_output, i)
        if gen_type == "bgm":
            codes_BKT[:, 1] = _VOCAL_REPLACEMENT
        elif gen_type == "vocal":
            codes_BKT[:, 2] = _BGM_REPLACEMENT
        elif gen_type != "mixed":
            logger.warning(
                "lelm_to_flow1dvae: unrecognised gen_type=%r for request %d; treating as 'mixed'",
                gen_type,
                i,
            )

        # Save raw codes for diagnostic comparison if requested.
        if _SAVE_CODES_PATH:
            save_data = {"codes_raw": codes_BKT.clone().cpu(), "gen_type": gen_type}
            torch.save(save_data, _SAVE_CODES_PATH)
            logger.info("Saved codes [%s] to %s", tuple(codes_BKT.shape), _SAVE_CODES_PATH)

        # Clamp any residual EOS/special tokens to valid codebook range.
        codes_BKT = codes_BKT.clamp(max=_VALID_CODE_MAX)

        # Stream-major flat layout: [3 * num_frames].
        codec_codes = codes_BKT.transpose(0, 1).contiguous().cpu().to(torch.long).reshape(-1).tolist()

        # Pass through optional prompt-audio waveforms and any per-request
        # metadata. The decoder reads these from
        # ``runtime_additional_information[i]`` in its ``forward()``.
        additional_information = _build_additional_information(prompt, lelm_output, i, gen_type=gen_type)

        flow1dvae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )

    return flow1dvae_inputs


# ---------------------------------------------------------------------------
# Async path: chunked streaming
# ---------------------------------------------------------------------------


def lelm_to_flow1dvae_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit codec chunks as the LeLM produces them (streaming path).

    Accumulates per-step [K=3] frames, then dispatches a window of
    `chunk_size + left_context_size` frames stream-major to the decoder
    at chunk boundaries.

    Default alignment: `codec_chunk_frames=750`, `codec_left_context_frames=250`
    (30 s audio per chunk with 10 s overlap at 25 fps). Deploy YAML can override.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Accumulate one frame per step from the pooling output.
    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach().to(device="cpu", dtype=torch.long))
    elif not finished:
        return None

    # Chunk sizes come from the connector config (deploy YAML).
    chunk_size, left_context_size_config, initial_chunk_size = _read_chunk_config(
        transfer_manager,
        request,
        default_chunk=750,
        default_left=250,
        default_initial=0,
    )
    if initial_chunk_size > chunk_size:
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    in_initial_phase = initial_chunk_size > 0 and length <= chunk_size

    if in_initial_phase:
        already_sent = transfer_manager.put_req_chunk[request_id] * initial_chunk_size
        pending = length - already_sent
        if pending <= 0:
            return None
        if pending < initial_chunk_size and not finished:
            return None
        context_length = min(pending, initial_chunk_size)
        left_context_size = max(0, length - context_length)
        window_frames = transfer_manager.code_prompt_token_ids[request_id][:length]
    else:
        initial_coverage = (chunk_size // initial_chunk_size) * initial_chunk_size if initial_chunk_size > 0 else 0
        adjusted = length - initial_coverage
        chunk_length = adjusted % chunk_size
        if chunk_length != 0 and not finished:
            return None
        context_length = chunk_length if chunk_length != 0 else chunk_size
        end_index = min(length, left_context_size_config + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Stack per-step [K] frames -> [num_frames, K] -> stream-major flat.
    stacked_frames = torch.stack(window_frames, dim=0)
    codec_codes = stacked_frames.transpose(0, 1).contiguous().reshape(-1)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codec_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_audio_codes(mm: dict[str, Any] | OmniPayload) -> torch.Tensor | None:
    """Find the [num_frames, K] codec tensor in the multimodal payload.

    Accepts both `mm["audio_codes"]` and `mm["codes"]["audio"]` layouts.
    """
    if not isinstance(mm, dict):
        return None
    direct = mm.get("audio_codes")
    if isinstance(direct, torch.Tensor) and direct.numel() > 0:
        return direct.to(torch.long)
    codes_val = mm.get("codes")
    if isinstance(codes_val, dict):
        nested = codes_val.get("audio")
        if isinstance(nested, torch.Tensor) and nested.numel() > 0:
            return nested.to(torch.long)
    return None


def _valid_frame_mask(codes_bkt: torch.Tensor) -> torch.Tensor:
    """Mark frames that carry at least one valid RVQ index in any stream."""
    return (codes_bkt <= _VALID_CODE_MAX).any(dim=1)


def _empty_omni_tokens_prompt():
    from vllm_omni.inputs.data import OmniTokensPrompt

    return OmniTokensPrompt(
        prompt_token_ids=[],
        multi_modal_data=None,
        mm_processor_kwargs=None,
        additional_information=None,
    )


def _gen_type_from_entry(entry: Any) -> str | None:
    """Pull gen_type from a prompt dict (top-level or additional_information)."""
    if isinstance(entry, dict):
        v = entry.get("gen_type") or (entry.get("additional_information") or {}).get("gen_type")
        if v is not None:
            return str(v)
    return None


def _resolve_gen_type(prompt: Any, lelm_output: Any, idx: int) -> str:
    """Resolve gen_type from request or per-prompt metadata."""
    # Per-request metadata on the RequestOutput.
    ai = getattr(lelm_output, "additional_information", None)
    if isinstance(ai, dict) and "gen_type" in ai:
        return str(ai["gen_type"])
    # The orchestrator forwards the original request prompt, which is a single
    # dict for one request or a list when batched. Handle both.
    if isinstance(prompt, list):
        if idx < len(prompt):
            v = _gen_type_from_entry(prompt[idx])
            if v is not None:
                return v
    else:
        v = _gen_type_from_entry(prompt)
        if v is not None:
            return v
    return "mixed"


def _build_additional_information(
    prompt: Any,
    lelm_output: Any,
    idx: int,
    *,
    gen_type: str,
) -> dict[str, Any] | None:
    """Pack per-request metadata that the Stage-1 decoder reads.

    Stage 1 forward() reads `runtime_additional_information[i]["meta"]["gen_type"]`
    and `...[i]["prompt_vocal_wav"]` / `...[i]["prompt_bgm_wav"]`.
    """
    prompt_vocal = _lookup_field(prompt, lelm_output, idx, "prompt_vocal_wav")
    prompt_bgm = _lookup_field(prompt, lelm_output, idx, "prompt_bgm_wav")

    info: dict[str, Any] = {}
    if gen_type and gen_type != "mixed":
        info.setdefault("meta", {})["gen_type"] = gen_type
    if isinstance(prompt_vocal, torch.Tensor):
        info["prompt_vocal_wav"] = prompt_vocal
    if isinstance(prompt_bgm, torch.Tensor):
        info["prompt_bgm_wav"] = prompt_bgm

    return info or None


def _lookup_field(prompt: Any, lelm_output: Any, idx: int, key: str) -> Any:
    ai = getattr(lelm_output, "additional_information", None)
    if isinstance(ai, dict) and key in ai:
        return ai[key]
    if isinstance(prompt, list) and idx < len(prompt):
        entry = prompt[idx]
        if isinstance(entry, dict):
            v = entry.get(key)
            if v is not None:
                return v
            ai2 = entry.get("additional_information")
            if isinstance(ai2, dict):
                return ai2.get(key)
    return None


def _extract_last_frame(pooling_output: dict[str, Any] | OmniPayload) -> torch.Tensor | None:
    """Return the latest single [K=3] frame from a per-step pooling output."""
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        codes_val = pooling_output.get("codes") if isinstance(pooling_output, dict) else None
        audio_codes = codes_val.get("audio") if isinstance(codes_val, dict) else None
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not (frame <= _VALID_CODE_MAX).any():
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        if not (audio_codes <= _VALID_CODE_MAX).any():
            return None
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for SongGeneration v2 async_chunk: {tuple(audio_codes.shape)}")


def _read_chunk_config(
    transfer_manager: Any,
    request: Any,
    *,
    default_chunk: int,
    default_left: int,
    default_initial: int,
) -> tuple[int, int, int]:
    """Read codec_chunk / left_context / initial_chunk from the connector config."""
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", default_chunk))
    left_context = int(cfg.get("codec_left_context_frames", default_left))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames", default_initial))

    ai = getattr(request, "additional_information", None)
    if ai is not None and hasattr(ai, "entries") and "initial_codec_chunk_frames" in ai.entries:
        entry = ai.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk = int(entry.list_data[0])

    if chunk_size <= 0 or left_context < 0 or initial_chunk < 0:
        raise ValueError(
            f"Invalid SongGeneration v2 chunk config: chunk={chunk_size}, "
            f"left_context={left_context}, initial={initial_chunk}"
        )
    return chunk_size, left_context, initial_chunk


__all__ = [
    "lelm_to_flow1dvae",
    "lelm_to_flow1dvae_async_chunk",
]
