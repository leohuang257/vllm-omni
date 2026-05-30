# SongGeneration v2-large — offline inference

Offline inference demo for the SongGeneration v2-large two-stage pipeline
via the vLLM-Omni `Omni()` runtime.

- **Stage 0**: LeLM AR (Llama-style 36L + 12L sub) producing 3-stream codec tokens
- **Stage 1**: Flow1dVAE diffusion decoder producing 48 kHz stereo audio

## Prerequisites

### 1. Clone the upstream repo

Stage 1 wraps the upstream Flow1dVAE/Tango source code. You must have a local
clone of the official SongGeneration repository:

```bash
git clone https://github.com/tencent-ailab/SongGeneration.git
```

Model weights and runtime assets (checkpoints, tokenizer) are **downloaded
automatically** on first run from HuggingFace. No manual download needed.

### 2. Install upstream Python dependencies

Stage 1 imports the upstream audio stack at runtime, so the following PyPI
packages must be available in your vLLM-Omni environment:

```bash
pip install \
  ninja alias-free-torch descript-audio-codec einops-exts flashy \
  nnAudio openunmix x-transformers vector-quantize-pytorch \
  k-diffusion julius librosa kaldiio lameenc
```

> **Do not** run `pip install -r SongGeneration/requirements.txt` directly.
> That file pins `torch==2.6.0`, `transformers==4.37.2`, `diffusers==0.27.2`,
> and `huggingface-hub==0.25.2`, which conflict with and would downgrade the
> versions vLLM-Omni requires. Install only the leaf audio packages above.

### 3. Add a root `config.json`

Omni resolves the model type from a `config.json` at the repo root, but the
upstream repo / HuggingFace snapshots do not ship one. Create it once:

```bash
cat > /path/to/SongGeneration/config.json <<'EOF'
{
  "model_type": "songgeneration_v2",
  "architectures": ["SongGenerationV2LeLMForConditionalGeneration"]
}
EOF
```

Without it, `Omni(...)` fails with `Could not determine model_type`.

## Quick Start

```bash
# Option 1: Pass path directly
python end2end.py --model /path/to/SongGeneration --query-type mixed

# Option 2: Set environment variable
export SONGGENERATION_REPO=/path/to/SongGeneration
python end2end.py --query-type mixed

# Option 3: Clone to a common path (auto-detected)
#   /root/SongGeneration, /workspace/SongGeneration, or ~/SongGeneration
python end2end.py --query-type mixed
```

On first run, the script downloads missing assets:
- `lglg666/SongGeneration-v2-large` → `songgeneration_v2_large/` (LeLM weights)
- `lglg666/SongGeneration-Runtime` → `ckpt/`, `third_party/` (decoder + tokenizer)

Use `--no-auto-download` to disable this behavior (offline environments).

## Query Types

| Type | Description |
|------|-------------|
| `mixed` | Full song with vocals + instrumental (default) |
| `vocal` | Vocal-only track |
| `bgm` | Instrumental-only track |

Each query type includes a built-in sample lyric with Chinese lyrics and
structure tags. Override with `--lyric` and `--descriptions`.

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | Auto-detect | Path to local SongGeneration repo |
| `--query-type` | `mixed` | `mixed`, `vocal`, or `bgm` |
| `--lyric` | Built-in sample | Override lyrics (see official format guide) |
| `--descriptions` | Built-in sample | Comma-separated style tags (e.g. `female, pop, sad, piano`) |
| `--seed` | `42` | AR sampling seed |
| `--max-gen-len` | `750` | Max AR frames (~20s at 25fps) |
| `--duration-sec` | — | Target audio duration (overrides max-gen-len) |
| `--prompt-len` | Auto | Override condition prefix length |
| `--output-dir` | `output_songgeneration_v2` | Output directory |
| `--no-auto-download` | off | Disable HuggingFace auto-download |

## Output

Produces 48 kHz stereo WAV files:

```
output_songgeneration_v2/output_<request_id>.wav
```

## Architecture

Uses `vllm_omni.Omni()` — the same runtime that serves online requests.
Stage 0 runs as `LLM_AR` (per-step autoregressive); Stage 1 runs as
`LLM_GENERATION` (single-shot decode via upstream Tango wrapper).

Stage 1 wraps upstream Flow1dVAE/Tango runtime assets. Weights and code
are loaded at runtime from the local SongGeneration repo; vLLM-Omni does
not vendor upstream weights. A native port can replace the wrapper later
without changing the stage contract.
