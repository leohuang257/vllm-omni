# DreamX-World-5B-Cam

> Image + caption + camera/action-controlled video generation (Wan2.2 TI2V-5B + PRoPE)

## Summary

- Vendor: GD-ML (AMAP)
- Model: `GD-ML/DreamX-World-5B-Cam`
- Task: Image-to-video world generation with explicit 6-DoF camera/action control
- Mode: Offline generation via the shared `image_to_video` example / `Omni` API
- Maintainer: Community

## When to use this recipe

Use this to generate camera-controllable videos from a single start image + a
caption + a sequence of camera action commands. The pipeline class is
`WanCameraPipeline`: a Wan2.2 TI2V-5B image-to-video backbone with a per-block
PRoPE camera self-attention branch.

The released checkpoint is **transformer-only**; the VAE / text-encoder /
tokenizer load from the base `Wan-AI/Wan2.2-TI2V-5B-Diffusers` by default
(override via `model_config["base_model_path"]`).

Camera action tokens (composable, e.g. `"wj"` = push in + pan left):

| Token | Control | Token | Control |
|-------|---------|-------|---------|
| `w` | push in | `s` | pull out |
| `a` | move left | `d` | move right |
| `i` | tilt up | `k` | tilt down |
| `j` | pan left | `l` | pan right |

`action_seq` / `action_speed_list` are declared in
[`vllm_omni/model_extras/dreamx_world.py`](../../vllm_omni/model_extras/dreamx_world.py)
and passed via `--extra-body`.

## References

- Model: https://huggingface.co/GD-ML/DreamX-World-5B-Cam
- Upstream: https://github.com/AMAP-ML/DreamX-World
- Base model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- Integration issue: https://github.com/vllm-project/vllm-omni/issues/4570

## Hardware Support

## GPU

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment with one H100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from
- Install: editable install from the vLLM-Omni repo root (`pip install -e .`); see the repository README for diffusion extras
- Hugging Face access: `GD-ML/DreamX-World-5B-Cam` plus base `Wan-AI/Wan2.2-TI2V-5B-Diffusers` are fetched on first run

#### Command

Reproduces the second item in upstream
[`configs/dreamx/eval.json`](https://github.com/AMAP-ML/DreamX-World/blob/master/configs/dreamx/eval.json)
(`demo/007.jpg`, `w` → `wj`). Clone the upstream repo for the start frame. The
model id auto-detects `WanCameraPipeline` (no `--model-class-name` needed).

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model GD-ML/DreamX-World-5B-Cam \
  --image /path/to/DreamX-World/demo/007.jpg \
  --prompt "Style: Minecraft. A serene Minecraft landscape at sunset, featuring a blocky cliffside overlooking a calm ocean. In the foreground, grassy terrain with yellow flowers and red soil leads up to a rugged cliff composed of layered red and gray blocks. Sparse trees grow on rocky outcrops, adding life to the structured environment. The midground reveals the cliff's dramatic descent into the water, while the background showcases a vast ocean reflecting the warm hues of the setting sun. The sky is painted in gradients of orange, pink, and pale blue, with pixelated clouds drifting above. The lighting casts soft shadows and enhances the textured, cubic surfaces, creating a peaceful and immersive atmosphere that blends natural beauty with digital artistry." \
  --height 704 --width 1280 --num-frames 121 --fps 24 \
  --num-inference-steps 50 --guidance-scale 3.0 --flow-shift 3.0 --seed 42 \
  --extra-body '{"action_seq": ["w", "wj"], "action_speed_list": [4, 6]}' \
  --output dreamx_i2v.mp4
```

#### Verification

A non-empty 704×1280, 121-frame MP4 is written to `dreamx_i2v.mp4`, with camera
motion following the action sequence. For numerical fidelity, compare latents
against the upstream `inference_dreamx_5b.sh` on the same `eval.json` item and seed.

#### Notes

- `num_frames` must satisfy the 1+4k pattern (e.g. 81, 121); it is snapped
  automatically. `121` frames = 5s @ 24fps; `81` = 5s @ 16fps.
- Camera control is **required**: `action_seq` + `action_speed_list` must be
  provided. The pipeline raises if they are missing (use the base `WanPipeline`
  for plain image-to-video).
- The long-horizon autoregressive `DreamX-World-5B` model is out of scope.
