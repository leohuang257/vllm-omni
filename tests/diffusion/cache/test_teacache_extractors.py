# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TeaCache extractor functions.

This module provides a generic testing framework for model-specific extractor functions
used by TeaCache. Each model's extractor can be tested by:
1. Creating a fixture that returns model module
2. Creating a fixture that returns sample inputs for that model
3. Creating a test class that inherits from BaseExtractorTest
4. Implementing any model-specific test methods

Currently implemented:
- TestFlux2KleinExtractor: Flux2Klein model extractor
- TestFlux2Extractor: Flux2 model extractor
- TestFluxExtractor: Flux model extractor
- TestCosmos3Extractor: Cosmos3 VFM model extractor
"""

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.extractors import (
    extract_cosmos3_context,
    extract_flux2_context,
    extract_flux2_klein_context,
    extract_flux_context,
)
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel,
)

pytestmark = [pytest.mark.core_model]


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group():
    """Set up TP group for each test function"""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1):
        with patch("vllm.distributed.parallel_state.get_tp_group") as mock_get_tp_group:
            mock_tp_group = MagicMock()
            mock_tp_group.world_size = 1
            mock_get_tp_group.return_value = mock_tp_group
            yield


class BaseExtractorTest(ABC):
    """Base class for testing TeaCache extractors.

    Subclasses should implement:
    - get_extractor(): Return extractor function
    - get_module(): Return model module
    - get_sample_inputs(): Return sample inputs for model
    """

    @abstractmethod
    def get_extractor(self):
        """Return extractor function to test."""
        pass

    @abstractmethod
    def get_module(self):
        """Return model module instance."""
        pass

    @abstractmethod
    def get_sample_inputs(self):
        """Return sample inputs for model."""
        pass


class TestFlux2KleinExtractor(BaseExtractorTest):
    """Test extract_flux2_klein_context function."""

    def get_extractor(self):
        return extract_flux2_klein_context

    @pytest.fixture
    def flux2_klein_module(self):
        """Create a minimal Flux2Transformer2DModel for testing."""
        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        return model

    def get_module(self, flux2_klein_module):
        return flux2_klein_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux2Klein.

        Note: hidden_states uses in_channels=128 (default for Flux2Klein),
        not inner_dim=6144. The x_embedder projects from 128 -> 6144.
        encoder_hidden_states uses joint_attention_dim=15360 (model default),
        which then gets projected to inner_dim=6144 by context_embedder.
        """
        batch_size = 1
        img_seq_len = 1024
        txt_seq_len = 512
        in_channels = 128  # Model default in_channels
        txt_dim = 15360  # Model default joint_attention_dim

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 4)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 4)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_modulated_input_shape(self, flux2_klein_module, sample_inputs):
        """Test that modulated_input has correct shape matching the model's inner_dim.

        Note: After x_embedder projection, hidden_states are projected from
        in_channels (128) to inner_dim (6144), so modulated_input should match
        the projected shape, not the input shape.
        """
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        inner_dim = flux2_klein_module.inner_dim
        assert context.modulated_input.shape == (batch_size, img_seq_len, inner_dim)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_run_transformer_blocks_callable(self, flux2_klein_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_postprocess_callable(self, flux2_klein_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.postprocess)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_extra_states_contains_full_transformer(self, flux2_klein_module, sample_inputs):
        """Test that extra_states contains run_flux2_full_transformer_with_single."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        assert context.extra_states is not None
        assert "run_flux2_full_transformer_with_single" in context.extra_states
        assert callable(context.extra_states["run_flux2_full_transformer_with_single"])

    def test_without_guidance(self, flux2_klein_module, sample_inputs):
        """Test context extraction works without guidance (no CFG)."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux2_klein_context(flux2_klein_module, **inputs)

        assert context is not None
        assert context.temb is not None

    @pytest.mark.cpu
    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux2_klein_context(
                invalid_module,
                hidden_states=torch.randn(1, 1024, 6144),
                encoder_hidden_states=torch.randn(1, 512, 15360),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 1024, 4)),
                txt_ids=torch.randint(0, 64, (1, 512, 4)),
            )


class TestFlux2Extractor(BaseExtractorTest):
    """Test extract_flux2_context function."""

    def get_extractor(self):
        return extract_flux2_context

    @pytest.fixture
    def flux2_module(self):
        """Create a minimal Flux2Transformer2DModel for testing."""
        from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel

        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        return model

    def get_module(self, flux2_module):
        return flux2_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux2.

        Note: hidden_states uses in_channels=128 (default for Flux2),
        not inner_dim=6144. The x_embedder projects from 128 -> 6144.
        encoder_hidden_states uses joint_attention_dim=15360 (model default),
        which then gets projected to inner_dim=6144 by context_embedder.
        """
        batch_size = 1
        img_seq_len = 1024
        txt_seq_len = 512
        in_channels = 128  # Model default in_channels
        txt_dim = 15360  # Model default joint_attention_dim

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 4)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 4)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_modulated_input_shape(self, flux2_module, sample_inputs):
        """Test that modulated_input has correct shape matching the model's inner_dim.

        Note: After x_embedder projection, hidden_states are projected from
        in_channels (128) to inner_dim (6144), so modulated_input should match
        the projected shape, not the input shape.
        """
        context = extract_flux2_context(flux2_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        inner_dim = flux2_module.inner_dim
        assert context.modulated_input.shape == (batch_size, img_seq_len, inner_dim)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_run_transformer_blocks_callable(self, flux2_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux2_context(flux2_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_postprocess_callable(self, flux2_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux2_context(flux2_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_without_guidance(self, flux2_module, sample_inputs):
        """Test context extraction works without guidance (no CFG)."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux2_context(flux2_module, **inputs)

        assert context is not None
        assert context.temb is not None

    @pytest.mark.cpu
    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux2_context(
                invalid_module,
                hidden_states=torch.randn(1, 1024, 6144),
                encoder_hidden_states=torch.randn(1, 512, 15360),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 1024, 4)),
                txt_ids=torch.randint(0, 64, (1, 512, 4)),
            )


@pytest.mark.cpu
class TestFluxExtractor(BaseExtractorTest):
    """Test extract_flux_context function."""

    @pytest.fixture(autouse=True)
    def cpu_vllm_config(self):
        """Force CPU custom-op dispatch for this test class."""
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
            yield

    @pytest.fixture(autouse=True)
    def mock_flux_attention_backend(self):
        """Use the SDPA backend so FLUX can be instantiated in CPU tests."""
        from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

        with patch(
            "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
            return_value=(SDPABackend, None),
        ):
            yield

    def get_extractor(self):
        return extract_flux_context

    @pytest.fixture
    def flux_module(self):
        """Create a minimal FluxTransformer2DModel for testing."""
        return FluxTransformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            axes_dims_rope=(4, 4, 8),
        )

    @pytest.fixture
    def flux_module_without_guidance(self):
        """Create a minimal non-guidance-distilled FLUX transformer."""
        return FluxTransformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(4, 4, 8),
        )

    def get_module(self, flux_module):
        return flux_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux."""
        batch_size = 1
        img_seq_len = 16
        txt_seq_len = 8
        in_channels = 64  # Flux default in_channels
        txt_dim = 32
        pooled_dim = 16

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "pooled_projections": torch.randn(batch_size, pooled_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 3)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 3)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, flux_module, sample_inputs):
        """Test that modulated_input has the projected FLUX inner dimension."""
        context = extract_flux_context(flux_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        assert context.modulated_input.shape == (batch_size, img_seq_len, flux_module.inner_dim)

    def test_run_transformer_blocks_callable(self, flux_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux_context(flux_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, flux_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux_context(flux_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_postprocess_output_shape(self, flux_module, sample_inputs):
        """Test that postprocess projects back to the input channel width."""
        context = extract_flux_context(flux_module, **sample_inputs)
        output = context.postprocess(context.hidden_states)

        assert output.sample.shape == sample_inputs["hidden_states"].shape

    def test_postprocess_return_tuple_when_return_dict_false(self, flux_module, sample_inputs):
        """Test that postprocess honors return_dict=False."""
        context = extract_flux_context(flux_module, **sample_inputs, return_dict=False)
        output = context.postprocess(context.hidden_states)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert output[0].shape == sample_inputs["hidden_states"].shape

    def test_without_guidance(self, flux_module_without_guidance, sample_inputs):
        """Test context extraction works for FLUX variants without guidance embeddings."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux_context(flux_module_without_guidance, **inputs)

        assert context is not None
        assert context.temb is not None

    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux_context(
                invalid_module,
                hidden_states=torch.randn(1, 16, 64),
                encoder_hidden_states=torch.randn(1, 8, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 16, 3)),
                txt_ids=torch.randint(0, 64, (1, 8, 3)),
            )


@pytest.mark.cpu
class TestCosmos3Extractor(BaseExtractorTest):
    """Test extract_cosmos3_context function."""

    @pytest.fixture(autouse=True)
    def cpu_vllm_config(self):
        """Force CPU custom-op dispatch for this test class."""
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
            yield

    @pytest.fixture(autouse=True)
    def mock_cosmos3_attention_backend(self):
        """Use the SDPA backend so Cosmos3 can be instantiated in CPU tests."""
        from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

        with patch(
            "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
            return_value=(SDPABackend, None),
        ):
            yield

    @pytest.fixture(autouse=True)
    def mock_cosmos3_parallel_state(self, monkeypatch):
        """Run single-rank: no TP and no Ulysses sequence parallelism."""
        from vllm.model_executor.layers import linear as vllm_linear

        from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

        monkeypatch.setattr(transformer_cosmos3, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))
        monkeypatch.setattr(vllm_linear, "get_tensor_model_parallel_rank", lambda: 0)

    def get_extractor(self):
        return extract_cosmos3_context

    @pytest.fixture
    def cosmos3_module(self):
        """Create a minimal Cosmos3VFMTransformer for testing."""
        from types import SimpleNamespace

        from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

        config = {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 32,
            "latent_patch_size": 1,
            "latent_channel": 2,
            "rope_scaling": {"mrope_section": [1, 1, 0]},
        }
        model = Cosmos3VFMTransformer(SimpleNamespace(tf_model_config=config, dtype=torch.float32))
        # vLLM parallel linear layers allocate uninitialized (torch.empty)
        # weights and expect a checkpoint load; fill all parameters with small
        # deterministic values so the tiny test model produces finite outputs.
        torch.manual_seed(42)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(mean=0.0, std=0.02)
        model.eval()
        return model

    def get_module(self, cosmos3_module):
        return cosmos3_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Cosmos3 (video-only T2V path)."""
        torch.manual_seed(0)
        return {
            "hidden_states": torch.randn(1, 2, 1, 2, 2),
            "timestep": torch.tensor([1000.0]),
            "text_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "text_mask": torch.ones(1, 3, dtype=torch.long),
            "video_shape": (1, 2, 2),
            "fps": 24.0,
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, cosmos3_module, sample_inputs):
        """Test that modulated_input covers the GEN token sequence at hidden_size width."""
        context = extract_cosmos3_context(cosmos3_module, **sample_inputs)

        t, h, w = sample_inputs["video_shape"]
        s_video = t * h * w  # latent_patch_size=1
        assert context.modulated_input.shape == (1, s_video, cosmos3_module.hidden_size)
        assert context.encoder_hidden_states is None

    def test_und_kv_cache_populated_and_reused(self, cosmos3_module, sample_inputs):
        """Test that the UND K/V cache is computed once and reused on later calls."""
        assert cosmos3_module.cached_kv is None
        extract_cosmos3_context(cosmos3_module, **sample_inputs)
        assert cosmos3_module.cached_kv is not None
        assert len(cosmos3_module.cached_kv) == len(cosmos3_module.gen_layers)

        cached_kv_before = cosmos3_module.cached_kv
        und_calls = []
        original_und_forward = cosmos3_module.language_model.forward

        def counting_forward(*args, **kwargs):
            und_calls.append(1)
            return original_und_forward(*args, **kwargs)

        cosmos3_module.language_model.forward = counting_forward
        extract_cosmos3_context(cosmos3_module, **sample_inputs)
        assert not und_calls
        assert cosmos3_module.cached_kv is cached_kv_before

    def test_full_compute_matches_direct_forward(self, cosmos3_module, sample_inputs):
        """Test that extractor preprocessing + blocks + postprocess equals module.forward."""
        with torch.no_grad():
            reference = cosmos3_module(**sample_inputs)

            cosmos3_module.reset_cache()
            context = extract_cosmos3_context(cosmos3_module, **sample_inputs)
            output = context.postprocess(context.run_transformer_blocks()[0])

        torch.testing.assert_close(output, reference)

    def test_teacache_hook_full_compute_parity(self, cosmos3_module, sample_inputs):
        """Test the hook end-to-end: with a never-cache threshold the wrapped
        forward must reproduce the original forward at every denoising step."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook

        timesteps = [1000.0, 900.0, 800.0]
        references = []
        with torch.no_grad():
            for t in timesteps:
                references.append(cosmos3_module(**{**sample_inputs, "timestep": torch.tensor([t])}))

        cosmos3_module.reset_cache()
        # Polynomial rescaling of any positive distance stays >= threshold,
        # so the hook computes the full transformer at every step.
        apply_teacache_hook(
            cosmos3_module,
            TeaCacheConfig(transformer_type="Cosmos3VFMTransformer", rel_l1_thresh=1e-9),
        )
        with torch.no_grad():
            for t, reference in zip(timesteps, references):
                output = cosmos3_module(**{**sample_inputs, "timestep": torch.tensor([t])})
                torch.testing.assert_close(output, reference)

    def test_teacache_hook_cached_step_reuses_residual(self, cosmos3_module, sample_inputs):
        """Test that a huge threshold makes later steps skip the GEN layers."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook

        apply_teacache_hook(
            cosmos3_module,
            TeaCacheConfig(transformer_type="Cosmos3VFMTransformer", rel_l1_thresh=1e9),
        )
        gen_layer_calls = []
        original_layer_forward = cosmos3_module.gen_layers[0].forward

        def counting_forward(*args, **kwargs):
            gen_layer_calls.append(1)
            return original_layer_forward(*args, **kwargs)

        cosmos3_module.gen_layers[0].forward = counting_forward

        with torch.no_grad():
            output_first = cosmos3_module(**sample_inputs)
            assert len(gen_layer_calls) == 1  # first step always computes
            output_second = cosmos3_module(**{**sample_inputs, "timestep": torch.tensor([900.0])})
            assert len(gen_layer_calls) == 1  # second step served from cache

        assert output_first.shape == sample_inputs["hidden_states"].shape
        assert output_second.shape == sample_inputs["hidden_states"].shape

    def test_cosmos3_hook_warmup_steps_always_compute(self, cosmos3_module, sample_inputs):
        """Test that the warmup window forces computation even at a huge threshold:
        steps 0-1 must compute, step 2 may then be served from cache."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
        from vllm_omni.diffusion.hooks import HookRegistry
        from vllm_omni.diffusion.models.cosmos3.teacache_cosmos3 import Cosmos3TeaCacheHook

        hook = Cosmos3TeaCacheHook(
            TeaCacheConfig(transformer_type="Cosmos3VFMTransformer", rel_l1_thresh=1e9),
            num_warmup_steps=2,
        )
        HookRegistry.get_or_create(cosmos3_module).register_hook(TeaCacheHook._HOOK_NAME, hook)
        gen_layer_calls = []
        original_layer_forward = cosmos3_module.gen_layers[0].forward

        def counting_forward(*args, **kwargs):
            gen_layer_calls.append(1)
            return original_layer_forward(*args, **kwargs)

        cosmos3_module.gen_layers[0].forward = counting_forward

        with torch.no_grad():
            for step, t in enumerate([1000.0, 900.0, 800.0]):
                cosmos3_module(**{**sample_inputs, "timestep": torch.tensor([t])})
                expected = min(step + 1, 2)  # steps 0-1 compute, step 2 cached
                assert len(gen_layer_calls) == expected

    def test_cosmos3_enabler_applies_warmup_default(self, cosmos3_module):
        """Test that the Cosmos3 enabler defaults the warmup window and honors overrides."""
        from types import SimpleNamespace

        from vllm_omni.diffusion.cache.teacache.backend import enable_cosmos3_teacache
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
        from vllm_omni.diffusion.data import DiffusionCacheConfig
        from vllm_omni.diffusion.models.cosmos3.teacache_cosmos3 import (
            COSMOS3_DEFAULT_NUM_WARMUP_STEPS,
            Cosmos3TeaCacheHook,
        )

        pipeline = SimpleNamespace(transformer=cosmos3_module)
        enable_cosmos3_teacache(pipeline, DiffusionCacheConfig())
        hook = cosmos3_module._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
        assert isinstance(hook, Cosmos3TeaCacheHook)
        assert hook.num_warmup_steps == COSMOS3_DEFAULT_NUM_WARMUP_STEPS > 0
        assert pipeline._cache_backend_requires_paired_cfg is True

        # Overrides arrive through DiffusionCacheConfig extra params
        # (the dataclass has no such field; from_dict keeps unknown keys).
        enable_cosmos3_teacache(pipeline, DiffusionCacheConfig.from_dict({"num_warmup_steps": 0}))
        hook = cosmos3_module._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
        assert hook.num_warmup_steps == 0

    def test_invalid_module_raises_error(self):
        """Test that invalid module without gen_layers raises ValueError."""
        invalid_module = Mock()
        invalid_module.gen_layers = []

        with pytest.raises(ValueError, match="Module must have gen_layers"):
            extract_cosmos3_context(
                invalid_module,
                hidden_states=torch.randn(1, 2, 1, 2, 2),
                timestep=torch.tensor([1000.0]),
                text_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                text_mask=torch.ones(1, 3, dtype=torch.long),
                video_shape=(1, 2, 2),
            )
