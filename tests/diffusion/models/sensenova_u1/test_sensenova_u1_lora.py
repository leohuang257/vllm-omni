# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SenseNova-U1 LoRA support (denoising stack under language_model, dual MoT branches)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.diffusion.lora.helpers import (
    DummyBaseLayerWithLoRA,
    FakeLinearBase,
    fake_replace_submodule,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Real mapping persisted by SenseNovaU1Pipeline.load_weights(). The gen-path MLP
# is a separate module (mlp_mot_gen) reusing the gate_up_proj leaf name, so there
# is intentionally no gate_up_proj_mot_gen entry.
_STACKED_PARAMS_MAPPING = [
    (".qkv_proj_mot_gen", ".q_proj_mot_gen", "q"),
    (".qkv_proj_mot_gen", ".k_proj_mot_gen", "k"),
    (".qkv_proj_mot_gen", ".v_proj_mot_gen", "v"),
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]


def _make_manager(pipeline: torch.nn.Module) -> DiffusionLoRAManager:
    return DiffusionLoRAManager(
        pipeline=pipeline, device=torch.device("cpu"), dtype=torch.bfloat16, max_cached_adapters=1
    )


def _build_pipeline(*, declare_lora_components: bool) -> torch.nn.Module:
    """One MoT decoder layer: self_attn.{qkv,o}_proj[_mot_gen] + {mlp,mlp_mot_gen}.{gate_up,down}_proj."""
    self_attn = torch.nn.Module()
    self_attn.qkv_proj = FakeLinearBase()
    self_attn.o_proj = FakeLinearBase()
    self_attn.qkv_proj_mot_gen = FakeLinearBase()
    self_attn.o_proj_mot_gen = FakeLinearBase()

    mlp = torch.nn.Module()
    mlp.gate_up_proj = FakeLinearBase()
    mlp.down_proj = FakeLinearBase()
    mlp_mot_gen = torch.nn.Module()
    mlp_mot_gen.gate_up_proj = FakeLinearBase()
    mlp_mot_gen.down_proj = FakeLinearBase()

    layer = torch.nn.Module()
    layer.self_attn = self_attn
    layer.mlp = mlp
    layer.mlp_mot_gen = mlp_mot_gen

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer])
    lm = torch.nn.Module()
    lm.model = model

    pipeline = torch.nn.Module()
    pipeline.language_model = lm
    # SenseNovaU1Pipeline.load_weights() persists this on the pipeline itself.
    pipeline.stacked_params_mapping = list(_STACKED_PARAMS_MAPPING)
    if declare_lora_components:
        pipeline._lora_components = ["language_model"]
    return pipeline


def _patch_replacement(monkeypatch, replace_calls: list[str] | None = None) -> None:
    """Patch the manager so layer replacement runs on CPU without vLLM."""
    import vllm_omni.diffusion.lora.manager as manager_mod

    monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", DummyBaseLayerWithLoRA)

    def _fake_from_layer(*, layer, **_kwargs):
        return DummyBaseLayerWithLoRA(layer) if isinstance(layer, FakeLinearBase) else layer

    monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer)
    monkeypatch.setattr(
        manager_mod,
        "replace_submodule",
        lambda root, name, sub: fake_replace_submodule(root, name, sub, replace_calls),
    )


def _peft_helper(target_modules: list[str]):
    return type("_PH", (), {"r": 1, "target_modules": target_modules})()


# ---------------------------------------------------------------------------
# Component discovery + packed mapping derivation
# ---------------------------------------------------------------------------


class TestComponentDiscovery:
    def test_pipeline_declares_language_model_as_lora_component(self):
        """The real pipeline class opts language_model into LoRA scanning."""
        from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

        assert SenseNovaU1Pipeline._lora_components == ["language_model"]

    def test_packed_modules_mapping_derived_from_stacked_params(self):
        """Manager derives qkv/gate_up packed->sublayer mappings from stacked_params_mapping."""
        manager = _make_manager(_build_pipeline(declare_lora_components=True))

        mapping = manager._packed_modules_mapping
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["qkv_proj_mot_gen"] == ["q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen"]
        # One gate_up_proj entry covers both mlp and mlp_mot_gen (matching is by leaf name).
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]


# ---------------------------------------------------------------------------
# Opt-in / additive guarantee (does NOT affect other models)
# ---------------------------------------------------------------------------


class TestOptInIsAdditive:
    def test_language_model_not_scanned_without_lora_components(self, monkeypatch):
        """Without _lora_components a top-level language_model is never scanned (BAGEL stays unchanged)."""
        replace_calls: list[str] = []
        _patch_replacement(monkeypatch, replace_calls)

        pipeline = _build_pipeline(declare_lora_components=False)
        manager = _make_manager(pipeline)
        monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _m: ["q", "k", "v"])
        manager._replace_layers_with_lora(_peft_helper(["qkv_proj"]))

        assert replace_calls == []
        assert manager._lora_modules == {}
        assert isinstance(pipeline.language_model.model.layers[0].self_attn.qkv_proj, FakeLinearBase)

    def test_default_components_still_scanned(self, monkeypatch):
        """Declaring _lora_components is purely additive; default names still work."""
        replace_calls: list[str] = []
        _patch_replacement(monkeypatch, replace_calls)

        pipeline = torch.nn.Module()
        pipeline.transformer = torch.nn.Module()
        pipeline.transformer.proj = FakeLinearBase()

        manager = _make_manager(pipeline)
        manager._replace_layers_with_lora(_peft_helper(["proj"]))

        assert "proj" in replace_calls
        assert "transformer.proj" in manager._lora_modules


# ---------------------------------------------------------------------------
# Layer replacement across BOTH MoT branches
# ---------------------------------------------------------------------------


class TestMoTBranchReplacement:
    def test_replaces_packed_attention_in_both_branches(self, monkeypatch):
        """Sublayer targets q_proj/q_proj_mot_gen replace the fused qkv layers in both branches."""
        _patch_replacement(monkeypatch)

        pipeline = _build_pipeline(declare_lora_components=True)
        manager = _make_manager(pipeline)
        monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _m: ["q", "k", "v"])
        manager._replace_layers_with_lora(_peft_helper(["q_proj", "q_proj_mot_gen"]))

        attn = pipeline.language_model.model.layers[0].self_attn
        assert isinstance(attn.qkv_proj, DummyBaseLayerWithLoRA)
        assert isinstance(attn.qkv_proj_mot_gen, DummyBaseLayerWithLoRA)
        # und/gen o_proj are not packed under qkv and must be left untouched.
        assert isinstance(attn.o_proj, FakeLinearBase)
        assert isinstance(attn.o_proj_mot_gen, FakeLinearBase)

    def test_replaces_gate_up_proj_in_both_mlp_branches(self, monkeypatch):
        """Targeting gate_up_proj wraps BOTH mlp and mlp_mot_gen (no _mot_gen suffix exists)."""
        _patch_replacement(monkeypatch)

        pipeline = _build_pipeline(declare_lora_components=True)
        manager = _make_manager(pipeline)
        manager._replace_layers_with_lora(_peft_helper(["gate_up_proj"]))

        layer = pipeline.language_model.model.layers[0]
        assert isinstance(layer.mlp.gate_up_proj, DummyBaseLayerWithLoRA)
        assert isinstance(layer.mlp_mot_gen.gate_up_proj, DummyBaseLayerWithLoRA)

    def test_replaces_single_row_parallel_layers_in_both_branches(self, monkeypatch):
        """Non-packed o_proj / down_proj are wrapped for und and gen paths alike."""
        _patch_replacement(monkeypatch)

        pipeline = _build_pipeline(declare_lora_components=True)
        manager = _make_manager(pipeline)
        manager._replace_layers_with_lora(_peft_helper(["o_proj", "o_proj_mot_gen", "down_proj"]))

        layer = pipeline.language_model.model.layers[0]
        assert isinstance(layer.self_attn.o_proj, DummyBaseLayerWithLoRA)
        assert isinstance(layer.self_attn.o_proj_mot_gen, DummyBaseLayerWithLoRA)
        assert isinstance(layer.mlp.down_proj, DummyBaseLayerWithLoRA)
        assert isinstance(layer.mlp_mot_gen.down_proj, DummyBaseLayerWithLoRA)


# ---------------------------------------------------------------------------
# Round-trip: synthetic checkpoint -> set_active_adapter -> verify weights
# ---------------------------------------------------------------------------


def _write_synthetic_lora(
    adapter_dir: Path, module_name: str, rank: int, in_dim: int, out_dim: int, b_fill: float = 2.0
) -> str:
    """Write a minimal LoRA adapter (safetensors + config) to *adapter_dir*.

    ``b_fill`` sets the constant value of lora_B so distinct adapters are
    distinguishable by the weights applied during activation.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": torch.ones((rank, in_dim)),
            f"base_model.model.{module_name}.lora_B.weight": torch.ones((out_dim, rank)) * b_fill,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [module_name]}), encoding="utf-8"
    )
    return str(adapter_dir)


class TestSenseNovaLoRARoundTrip:
    def test_set_active_adapter_loads_and_activates(self, tmp_path, monkeypatch):
        """Synthetic checkpoint -> load -> replace under language_model -> activate scaled weights."""
        _patch_replacement(monkeypatch)

        pipeline = _build_pipeline(declare_lora_components=True)
        manager = _make_manager(pipeline)

        module_name = "language_model.model.layers.0.self_attn.o_proj_mot_gen"
        lora_dir = _write_synthetic_lora(tmp_path / "lora", module_name, rank=2, in_dim=4, out_dim=4)
        lora_request = LoRARequest(lora_name="test_sensenova", lora_int_id=7, lora_path=lora_dir)

        manager.set_active_adapter(lora_request, lora_scale=0.5)

        replaced = pipeline.language_model.model.layers[0].self_attn.o_proj_mot_gen
        assert isinstance(replaced, DummyBaseLayerWithLoRA)
        assert len(replaced.set_calls) == 1
        lora_a, lora_b = replaced.set_calls[0]
        assert torch.all(lora_a == 1.0)
        assert torch.allclose(lora_b, torch.ones_like(lora_b))  # 2.0 * 0.5


# ---------------------------------------------------------------------------
# Dynamic adapter swapping at request time (A -> B -> A -> off)
# ---------------------------------------------------------------------------


class TestDynamicAdapterSwapping:
    def test_swap_between_distinct_adapters_per_request(self, tmp_path, monkeypatch):
        """Switching lora_request per request activates the right adapter, with no stale carryover."""
        _patch_replacement(monkeypatch)

        pipeline = _build_pipeline(declare_lora_components=True)
        manager = _make_manager(pipeline)  # max_cached_adapters=1 -> swaps evict + reload

        module_name = "language_model.model.layers.0.self_attn.o_proj_mot_gen"
        dir_a = _write_synthetic_lora(tmp_path / "a", module_name, rank=2, in_dim=4, out_dim=4, b_fill=2.0)
        dir_b = _write_synthetic_lora(tmp_path / "b", module_name, rank=2, in_dim=4, out_dim=4, b_fill=3.0)
        req_a = LoRARequest(lora_name="adapter_a", lora_int_id=1, lora_path=dir_a)
        req_b = LoRARequest(lora_name="adapter_b", lora_int_id=2, lora_path=dir_b)

        # request 1: adapter A
        manager.set_active_adapter(req_a, lora_scale=1.0)
        assert manager._active_adapter_id == 1
        # request 2: swap to adapter B
        manager.set_active_adapter(req_b, lora_scale=1.0)
        assert manager._active_adapter_id == 2
        # request 3: swap back to A (was evicted; must reload, not reuse stale B)
        manager.set_active_adapter(req_a, lora_scale=1.0)
        assert manager._active_adapter_id == 1
        # request 4: no adapter -> deactivate
        manager.set_active_adapter(None)
        assert manager._active_adapter_id is None

        # Fetch the wrapped layer after replacement; it persists across swaps so its
        # activations reflect the per-request sequence A, B, A.
        layer = pipeline.language_model.model.layers[0].self_attn.o_proj_mot_gen
        assert isinstance(layer, DummyBaseLayerWithLoRA)
        b_values = [float(b.flatten()[0]) for _, b in layer.set_calls]
        assert b_values == [2.0, 3.0, 2.0]
        assert layer.reset_calls >= 1  # deactivation reset the layer
